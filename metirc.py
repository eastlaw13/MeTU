import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
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
        for _ in range(30):
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
        "images_dir": Path("../../data/CityScapes/images/val"),  # 경로 확인 필요
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
# 3. 메인 평가 루프 (Dataset -> Model 순회)
# ==========================================
for ds_name, ds_info in EVAL_CONFIGS.items():
    print(f"\n{'='*20} DATASET: {ds_name} {'='*20}")

    # 해당 데이터셋의 transforms 임포트
    if ds_name == "VOC":
        from datasets.VOC2012 import Valtransforms

        transforms = Valtransforms()
    else:
        from datasets.CityScapes import ValTransforms

        # Cityscapes는 보통 crop_size를 명시하거나 하단 dummy_res를 따름
        transforms = ValTransforms(crop_size=(512, 1024))

    image_paths = sorted(ds_info["images_dir"].rglob(ds_info["img_ext"]))
    mask_paths = sorted(ds_info["masks_dir"].rglob("*.png"))

    for cfg in ds_info["models"]:
        print(f"\n>>> Evaluating {cfg['name']} on {ds_name}...")

        # 모델 로드
        try:
            model = cfg["class"].load_from_checkpoint(cfg["ckpt"]).to(device)
        except Exception as e:
            print(f"Failed to load {cfg['name']}: {e}")
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
        IoU_per_class = IoU_per_class.tolist()
        IoU_per_class = [round(num, 3) for num in IoU_per_class]

        # (2) Efficiency 측정
        dummy_input = torch.randn(*ds_info["dummy_res"]).to(device)
        flops_analyzer = FlopCountAnalysis(model, dummy_input)
        flops_analyzer.unsupported_ops_warnings(False)

        total_flops = flops_analyzer.total()
        total_params = sum(p.numel() for p in model.parameters())
        avg_lat, fps = measure_latency(model, dummy_input)

        # (3) 결과 출력
        print("-" * 50)
        print(f"[{ds_name} | {cfg['name']}] RESULTS")
        print("-" * 50)
        print(f"mIoU: {mIoU:.4f} | FPS: {fps:.2f} ({avg_lat:.2f} ms)")
        print(
            f"Params: {format_numbers(total_params)} | FLOPs: {format_numbers(total_flops)}"
        )
        print(f"Per-Class IoU:\n{IoU_per_class}")
        print("-" * 50)

        # 메모리 정리
        del model
        torch.cuda.empty_cache()

print("\nAll datasets and models have been evaluated.")
