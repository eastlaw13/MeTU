import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
import json
import os
from datetime import datetime
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

# 모델 및 유틸리티 임포트
from models.modelzoo import lt_MeTU, lt_segformerb0, lt_mobilevit_dlv3_p
from fvcore.nn import FlopCountAnalysis
from utils.iou import iou_component, iou_calculation

# fvcore 경고 숨기기 설정
logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)

# ==========================================
# 1. 전역 설정 및 유틸리티
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 결과 저장을 위한 디렉토리 및 파일명 설정
RESULTS_DIR = Path("./evaluation_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
TXT_PATH = RESULTS_DIR / f"eval_report_{TIMESTAMP}.txt"
JSON_PATH = RESULTS_DIR / f"eval_data_{TIMESTAMP}.json"


def format_numbers(n, decimals=3):
    if n >= 1e9:
        return f"{n / 1e9:.{decimals}f} G"
    elif n >= 1e6:
        return f"{n / 1e6:.{decimals}f} M"
    else:
        return f"{n:,}"


def measure_latency(model, dummy_input, iterations=100):
    model.eval()
    with torch.no_grad():
        for _ in range(30):  # Warmup
            _ = model(dummy_input)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))
    return np.mean(timings), 1000 / np.mean(timings)


def save_results(results_dict, txt_log_list):
    """
    결과를 JSON과 TXT 파일로 저장하는 함수.
    매 모델 평가가 끝날 때마다 호출하여 데이터 손실 방지.
    """
    # 1. JSON 저장
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)

    # 2. TXT 저장 (덮어쓰기 모드)
    with open(TXT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_log_list))

    print(f"✅ Results saved to {JSON_PATH} and {TXT_PATH}")


# ==========================================
# 2. 데이터셋별 구성 설정 (VOC & Cityscapes)
# ==========================================
EVAL_CONFIGS = {
    "VOC": {
        "images_dir": Path("../../data/VOC2012/images/val"),
        "masks_dir": Path("../../data/VOC2012/masks/val"),
        "total_class": 21,
        "dummy_res": (1, 3, 512, 512),
        "img_ext": "*.jpg",
        "models": [
            {
                "name": "MeTU-xxs",
                "class": lt_MeTU,
                "ckpt": "logs/VOC2012/1214/MeTU-xxs/best/BEST_mIoU_mIoU=0.606.ckpt",
            },
            {
                "name": "MeTU-xs",
                "class": lt_MeTU,
                "ckpt": "logs/VOC2012/1215/MeTU-xs/best/BEST_mIoU_mIoU=0.637.ckpt",
            },
            {
                "name": "SegFormer-B0",
                "class": lt_segformerb0,
                "ckpt": "logs/VOC2012/1215/Segformer-b0/best/BEST_mIoU_mIoU=0.623.ckpt",
            },
            {
                "name": "MobileViT-xxs-DLV3p",
                "class": lt_mobilevit_dlv3_p,
                "ckpt": "logs/VOC2012/1215/MobileViT-xxs/best/BEST_mIoU_mIoU=0.620.ckpt",
            },
            {
                "name": "MobileViT-xs-DLV3p",
                "class": lt_mobilevit_dlv3_p,
                "ckpt": "logs/VOC2012/1215/MobileViT-xs/best/BEST_mIoU_mIoU=0.642.ckpt",
            },
        ],
    },
    "Cityscapes": {
        "images_dir": Path("../../data/CityScapes/images/val"),
        "masks_dir": Path("../../data/CityScapes/masks/val"),
        "total_class": 19,
        "dummy_res": (1, 3, 512, 1024),
        "img_ext": "*.png",
        "models": [
            {
                "name": "MeTU-xxs",
                "class": lt_MeTU,
                "ckpt": "logs/Cityscapes/1124/MeTU-xxs/best/mIoU=0.709.ckpt",
            },
            {
                "name": "MeTU-xs",
                "class": lt_MeTU,
                "ckpt": "logs/Cityscapes/1125/MeTU-xs/best/mIoU=0.740.ckpt",
            },
            {
                "name": "SegFormer-B0",
                "class": lt_segformerb0,
                "ckpt": "logs/Cityscapes/1117/Segformer-b0/best/mIoU=0.704.ckpt",
            },
            {
                "name": "MobileViT-xxs-DLV3p",
                "class": lt_mobilevit_dlv3_p,
                "ckpt": "logs/Cityscapes/1208/MobileViT-xxs/best/mIoU=0.693.ckpt",
            },
            {
                "name": "MobileViT-xs-DLV3p",
                "class": lt_mobilevit_dlv3_p,
                "ckpt": "logs/Cityscapes/1209/MobileViT-xs/best/mIoU=0.727.ckpt",
            },
        ],
    },
}

# ==========================================
# 3. 메인 평가 루프
# ==========================================

# 전체 결과를 저장할 딕셔너리와 로그 리스트
full_results = {}
log_buffer = [f"Evaluation Report - Started at {TIMESTAMP}", "=" * 50]

for ds_name, ds_info in EVAL_CONFIGS.items():
    print(f"\n{'='*20} DATASET: {ds_name} {'='*20}")
    log_buffer.append(f"\n{'='*20} DATASET: {ds_name} {'='*20}")

    # 데이터셋 결과를 담을 딕셔너리 초기화
    full_results[ds_name] = {}

    # Transform 설정
    if ds_name == "VOC":
        from datasets.VOC2012 import Valtransforms

        transforms = Valtransforms()
    else:
        from datasets.CityScapes import ValTransforms

        transforms = ValTransforms(crop_size=(512, 1024))

    image_paths = sorted(ds_info["images_dir"].rglob(ds_info["img_ext"]))
    mask_paths = sorted(ds_info["masks_dir"].rglob("*.png"))

    for cfg in ds_info["models"]:
        print(f"\n>>> Evaluating {cfg['name']} on {ds_name}...")

        # 모델 로드
        try:
            model = cfg["class"].load_from_checkpoint(cfg["ckpt"]).to(device)
        except Exception as e:
            err_msg = f"Failed to load {cfg['name']}: {e}"
            print(err_msg)
            log_buffer.append(err_msg)
            continue

        model.eval()
        model.freeze()

        inter_total = torch.zeros(ds_info["total_class"], dtype=torch.long)
        union_total = torch.zeros(ds_info["total_class"], dtype=torch.long)

        # (1) mIoU 측정
        with torch.no_grad():
            for img_p, msk_p in tqdm(
                zip(image_paths, mask_paths),
                total=len(image_paths),
                desc=f"{cfg['name']} IoU",
            ):
                img = Image.open(img_p).convert("RGB")
                mask = Image.open(msk_p)
                img, gt_mask = transforms(img, mask)
                img_tensor = img.unsqueeze(0).to(device)

                logits = model(img_tensor)

                if logits.shape[-2:] != gt_mask.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=gt_mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                pred = logits.argmax(dim=1).squeeze().cpu()
                inter, union = iou_component(pred, gt_mask, ds_info["total_class"], 255)
                inter_total += inter
                union_total += union

        mIoU, IoU_per_class = iou_calculation(inter_total, union_total)
        # JSON 저장을 위해 Numpy 타입을 Python float으로 변환
        IoU_per_class = [round(float(num), 3) for num in IoU_per_class.tolist()]
        mIoU = float(mIoU)

        # (2) Efficiency 측정
        dummy_input = torch.randn(*ds_info["dummy_res"]).to(device)
        flops_analyzer = FlopCountAnalysis(model, dummy_input)
        flops_analyzer.unsupported_ops_warnings(False)

        total_flops = flops_analyzer.total()
        total_params = sum(p.numel() for p in model.parameters())
        avg_lat, fps = measure_latency(model, dummy_input)

        # 형 변환 (JSON 호환성)
        fps = float(fps)
        avg_lat = float(avg_lat)
        total_flops = int(total_flops)
        total_params = int(total_params)

        # (3) 결과 텍스트 생성
        result_str = (
            f"[{ds_name} | {cfg['name']}] RESULTS\n"
            f"-" * 50 + "\n"
            f"mIoU    : {mIoU:.4f}\n"
            f"FPS     : {fps:.2f} ({avg_lat:.2f} ms)\n"
            f"Params  : {format_numbers(total_params)}\n"
            f"FLOPs   : {format_numbers(total_flops)}\n"
            f"Per-Class IoU: {IoU_per_class}\n"
            f"-" * 50
        )

        # 콘솔 출력
        print("-" * 50)
        print(result_str)

        # 로그 버퍼에 추가
        log_buffer.append(result_str)

        # (4) 결과 딕셔너리에 저장 (Raw Data for JSON)
        full_results[ds_name][cfg["name"]] = {
            "mIoU": mIoU,
            "FPS": fps,
            "Latency_ms": avg_lat,
            "Params": total_params,
            "Params_str": format_numbers(total_params),
            "FLOPs": total_flops,
            "FLOPs_str": format_numbers(total_flops),
            "Per_Class_IoU": IoU_per_class,
        }

        # (5) 중간 저장 (실행 중 멈춰도 데이터 보존)
        save_results(full_results, log_buffer)

        # 메모리 정리
        del model
        torch.cuda.empty_cache()

print(f"\nEvaluation Finished. Results saved to:\n1. {JSON_PATH}\n2. {TXT_PATH}")
