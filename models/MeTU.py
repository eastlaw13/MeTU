import os
import sys
import timm
import wandb
import torch
import warnings
import shutil
import numpy as np
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchinfo import summary
from typing import Tuple, Dict
from transformers import logging

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.iou import iou_component, iou_calculation
from utils.Loss import DiceLoss, FocalLoss
from utils.lr_schedule import CosineAnnealingWithWarmupLR


warnings.filterwarnings("ignore")
logging.set_verbosity_warning()
logging.set_verbosity_error()


def generate_color_palette(num_classes: int) -> np.ndarray:
    palette = []

    for i in range(num_classes):
        hue = (i * 360) // num_classes

        # PIL 이미지 생성 및 RGB 변환
        rgb_img = Image.new("HSV", (1, 1), (hue, 255, 128)).convert("RGB")

        # 🚨 [수정 부분] 픽셀 값을 가져오되, 불필요한 * 255 곱셈을 제거하고 int로 변환만 합니다.
        r, g, b = [int(x) for x in rgb_img.getpixel((0, 0))]

        palette.extend([r, g, b])

    # 이 부분은 이전 단계에서 이미 수정되었고, 이제 정상 작동할 것입니다.
    palette_np = np.array(palette, dtype=np.uint8)
    full_palette = np.zeros(256 * 3, dtype=np.uint8)
    full_palette[: len(palette)] = palette_np

    return full_palette


def apply_color_map(mask_tensor: torch.Tensor, num_classes: int) -> Image.Image:
    """
    흑백 마스크 텐서를 컬러 이미지로 변환합니다.
    """
    # 텐서를 CPU NumPy 배열로 변환
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

    # 마스크 크기를 (H, W)로 설정
    mask_img = Image.fromarray(mask_np, mode="L")

    # 컬러 팔레트 생성 및 적용
    palette = generate_color_palette(num_classes)
    mask_img.putpalette(palette)

    # 'P' (팔레트) 모드를 'RGB'로 변환하여 시각화 준비
    return mask_img.convert("RGB")


class MobileViTEncoder(nn.Module):
    """
    Pretrained MobileViT v1 with Imagnet-1k based Encoder from timm API.

    Args:
        model_size (str): 'xxs', 'xs', 's'. Current verison only can use xxs ver.
        pretrained (bool): If True, load ImageNet pretrained weights
    Returns:
        model(nn.Module)
    """

    def __init__(self, model_size="xxs", pretrained=True):
        super().__init__()
        self.model_size = model_size
        self.pretrained = pretrained
        self.full_model = timm.create_model(
            f"mobilevit_{self.model_size}", pretrained=self.pretrained
        )
        self.stem = self.full_model.stem
        self.stage0 = self.full_model.stages[0]
        self.stage1 = self.full_model.stages[1]
        self.stage2 = self.full_model.stages[2]
        self.stage3 = self.full_model.stages[3]
        self.stage4 = self.full_model.stages[4]
        self.final_conv = self.full_model.final_conv

    def forward(self, x):

        skips_features = []
        out = self.stem(x)
        skips_features.append(out)
        out = self.stage0(out)
        skips_features.append(out)

        out = self.stage1(out)
        skips_features.append(out)

        out = self.stage2(out)
        skips_features.append(out)

        out = self.stage3(out)
        skips_features.append(out)

        out = self.stage4(out)
        skips_features.append(out)

        out = self.final_conv(out)

        return out, skips_features


class DWConv(nn.Module):
    """
    Depthwise separable Convoltuion block.

    Args:
        in_ch (int): Input channel size.
        out_ch (int): Output channel size.
        kernel (int): Kernel size. Default value is 3.
        padding (int): Padding size. Default value is 1.
        dilation (int): dilation value. Default value is 1.

    Returns:
        Depthwise convolution block (nn.Module)
    """

    def __init__(self, in_ch, out_ch, kernel=3, padding=1, dilation=1):
        super().__init__()
        effective_padding = padding * dilation
        self.depth = nn.Conv2d(
            in_ch,
            in_ch,
            kernel,
            padding=effective_padding,
            groups=in_ch,
            bias=False,
            dilation=dilation,
        )
        self.point = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, max(ch // r, 1), 1)
        self.fc2 = nn.Conv2d(max(ch // r, 1), ch, 1)

    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ImprovedDecoderBlock(nn.Module):
    """
    Improved decoder block that respects in_ch, skip_ch, out_ch.
    - Upsamples decoder feature x to match skip spatial size (if skip provided)
    - Projects skip to out_ch via 1x1
    - Concatenates [x, skip_proj] -> channel reduction via 1x1
    - Refinement via DWConv x2 + SE + residual (residual built from in_ch -> out_ch)
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        if skip_ch is None:
            self.upsample = nn.ConvTranspose2d(
                in_ch, in_ch, kernel_size=2, stride=2, bias=False
            )
        else:
            self.up_conv = nn.ConvTranspose2d(
                in_ch, in_ch, kernel_size=2, stride=2, bias=False
            )
            self.up_bn = nn.BatchNorm2d(in_ch)

        self.skip_proj = (
            nn.Conv2d(skip_ch, out_ch, kernel_size=1) if skip_ch is not None else None
        )

        self.reduce_decoder = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

        self.fuse_conv = nn.Conv2d(
            out_ch * (2 if skip_ch is not None else 1),
            out_ch,
            kernel_size=1,
            bias=False,
        )
        self.bn_fuse = nn.BatchNorm2d(out_ch)

        self.dw1 = DWConv(out_ch, out_ch, dilation=2, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.dw2 = DWConv(out_ch, out_ch, dilation=4, padding=1)
        self.act2 = nn.ReLU(inplace=True)

        self.se = SEBlock(out_ch, r=4)

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_res = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip=None):

        if skip is not None and self.skip_proj is not None:

            x_up = F.relu(self.up_bn(self.up_conv(x)), inplace=True)
            x_up = F.interpolate(
                x_up, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

            x_red = self.reduce_decoder(x_up)  # (B, out_ch, hs, ws)

            skip_p = self.skip_proj(skip)  # (B, out_ch, hs, ws)

            fused = torch.cat([x_red, skip_p], dim=1)  # (B, 2*out_ch, hs, ws)
            fused = self.bn_fuse(self.fuse_conv(fused))  # (B, out_ch, hs, ws)

            res_input = x_up
        else:
            x_up = F.relu(self.bn_fuse(self.upsample(x)), inplace=True)

            fused = self.reduce_decoder(x_up)
            fused = self.bn_fuse(self.fuse_conv(fused))

            res_input = x_up

        y = self.dw1(fused)
        y = self.act1(y)
        y = self.dw2(y)
        y = self.act2(y)

        y = self.se(y)

        res = self.res_conv(res_input)
        res = self.bn_res(res)

        out = F.relu(y + res, inplace=True)
        return out


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (CA) mechanism for efficient positional context encoding.
    (Memory efficient replacement for LTBlock's self-attention at high resolution)
    """

    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        _, _, h, w = x.size()

        # 1. Global Pooling (Horizontal and Vertical)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 2. Concatenation and Shared 1D Conv
        y = torch.cat([x_h, x_w], dim=2)
        y = self.bn1(self.conv1(y))
        y = self.act(y)

        # 3. Splitting and Separate 1D Conv
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 4. Final Gates
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        return identity * a_w * a_h


class DCNRefinement(nn.Module):
    """
    DWConv + Dilated Conv + Coordinate Attention Refinement Block.
    (Replaces LTBlock for high-resolution efficiency)
    """

    def __init__(self, ch):
        super().__init__()
        # 1. Local Feature Enhancement (DWConv)
        self.dw = DWConv(ch, ch, kernel=3)
        self.act1 = nn.ReLU(inplace=True)

        # 2. Multi-Scale Context (Dilated DWConv)
        # Dilated Conv를 사용하여 receptive field 확대 (Global Context 대체)
        self.d_conv = nn.Sequential(
            DWConv(ch, ch, dilation=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # 3. Global/Positional Context (Coordinate Attention)
        self.ca = CoordinateAttention(ch, ch)

        # 4. Fusion
        self.fuse_conv = nn.Conv2d(ch * 2, ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x):
        res = x

        # Branch 1: Local Feature + Positional Attention
        x_dw = self.act1(self.dw(x))
        x_ca = self.ca(x_dw)  # CA는 local feature에 집중도 부여

        # Branch 2: Multi-scale Feature
        x_d = self.d_conv(x)

        # Fusion
        fused = torch.cat([x_ca, x_d], dim=1)  # (B, 2*ch, H, W)
        out = F.relu(self.bn(self.fuse_conv(fused)) + res, inplace=True)
        return out


class ImprovedDecoderBlockV4(nn.Module):
    """
    Improved decoder block using DCNRefinement (DWConv + Dilated + CA) for high-res efficiency.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        # --- Up-sampling ---
        if skip_ch is None:
            self.upsample = nn.ConvTranspose2d(
                in_ch, in_ch, kernel_size=2, stride=2, bias=False
            )
            self.up_bn = nn.BatchNorm2d(in_ch)
        else:
            self.up_conv = nn.ConvTranspose2d(
                in_ch, in_ch, kernel_size=2, stride=2, bias=False
            )
            self.up_bn = nn.BatchNorm2d(in_ch)

        # --- Skip Projection / Feature Reduction / Fusion ---
        self.skip_proj = (
            nn.Conv2d(skip_ch, out_ch, kernel_size=1) if skip_ch is not None else None
        )
        self.reduce_decoder = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.fuse_conv = nn.Conv2d(
            out_ch * (2 if skip_ch is not None else 1),
            out_ch,
            kernel_size=1,
            bias=False,
        )
        self.bn_fuse = nn.BatchNorm2d(out_ch)

        # --- Refinement Block (DCNRefinement replaces LTBlock) ---
        self.refinement = DCNRefinement(ch=out_ch)

        # --- Residual Connection Path ---
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn_res = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip=None):

        # --- 1. Up-sampling and Interpolation & 2. Feature Fusion ---
        if skip is not None and self.skip_proj is not None:
            x_up = F.relu(self.up_bn(self.up_conv(x)), inplace=True)
            x_up = F.interpolate(
                x_up, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
            x_red = self.reduce_decoder(x_up)
            skip_p = self.skip_proj(skip)
            fused = torch.cat([x_red, skip_p], dim=1)
            fused = self.bn_fuse(self.fuse_conv(fused))
            res_input = x_up
        else:
            x_up = F.relu(self.up_bn(self.upsample(x)), inplace=True)
            fused = self.reduce_decoder(x_up)
            fused = self.bn_fuse(self.fuse_conv(fused))
            res_input = x_up

        # --- 3. Refinement via DCNRefinement ---
        y = self.refinement(fused)

        # --- 4. Final Residual Connection ---
        res = self.res_conv(res_input)
        res = self.bn_res(res)

        out = F.relu(y + res, inplace=True)
        return out


class MeTU(nn.Module):
    def __init__(self, model_size="xxs", encoder_pretrained=True, classes=1):
        super().__init__()
        self.encoder = MobileViTEncoder(model_size, encoder_pretrained)
        self.classes = classes

        # without final_conv, channels sizes: [stem, stage0, stage1, stage2, stage3, stage4]
        if model_size == "xxs":
            skip_channels = [16, 16, 24, 48, 64, 80]
        elif model_size == "xs":
            skip_channels = [16, 32, 48, 64, 80, 96]
        elif model_size == "s":
            skip_channels = [16, 32, 64, 96, 128, 160]
        else:
            e = print(
                "[WARNING] No matched size pretrained MobileViT model. Check the model_size input."
            )
            return e

        bottleneck_ch = self.encoder.final_conv.out_channels

        decoder_config = []
        prev_ch = bottleneck_ch
        for skip_ch in reversed(skip_channels):
            out_ch = max(skip_ch, prev_ch // 2)
            decoder_config.append((prev_ch, skip_ch, out_ch))
            prev_ch = out_ch

        self.decoder_blocks = nn.ModuleList(
            [
                ImprovedDecoderBlockV4(
                    in_ch=in_ch,
                    skip_ch=skip_ch,
                    out_ch=out_ch,
                )
                for in_ch, skip_ch, out_ch in decoder_config
            ]
        )

        self.out_conv = nn.Conv2d(decoder_config[-1][2], self.classes, kernel_size=1)

        for param in self.encoder.parameters():
            param.requires_grad = True

        # for name, param in self.encoder.named_parameters():
        #     if any(
        #         x in name
        #         for x in [
        #             "stem",
        #             "stages.0",
        #             "stages.1",
        #             "stages.2",
        #             "stages.3",
        #             "stages.4",
        #             "final_conv",
        #         ]
        #     ):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        # For update batch norms in shallow layer.
        self.encoder.stem.train()
        self.encoder.stage0.train()
        self.encoder.stage1.train()
        self.encoder.stage2.train()
        self.encoder.stage3.train()
        self.encoder.stage4.train()
        self.encoder.final_conv.train()

        # For freeze batch norms in deep layer.
        # self.encoder.stage4.eval()
        # self.encoder.final_conv.eval()

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        bottleneck, skips = self.encoder(x)
        x = bottleneck
        for i, dec_block in enumerate(self.decoder_blocks):
            skip_feat = skips[-(i + 1)]
            x = dec_block(x, skip_feat)

        x = F.interpolate(
            x, size=(input_h, input_w), mode="bilinear", align_corners=False
        )
        out = self.out_conv(x)
        return out


class lt_MeTU(L.LightningModule):
    def __init__(
        self,
        learning_rate,
        model_size="xxs",
        encoder_pretrained=True,
        classes=1,
    ):
        super().__init__()
        self.model_size = model_size
        self.encoder_pretrained = encoder_pretrained
        self.classes = classes
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self._init_iou_components()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.ignore_index = 255
        self.model = MeTU(
            model_size=self.model_size,
            encoder_pretrained=self.encoder_pretrained,
            classes=self.classes,
        )

    def _init_iou_components(self):
        num_classes = self.classes

        self.register_buffer(
            "train_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("train_unions", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer(
            "val_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("val_unions", torch.zeros(num_classes, dtype=torch.long))

        self.register_buffer(
            "test_intersections", torch.zeros(num_classes, dtype=torch.long)
        )
        self.register_buffer("test_unions", torch.zeros(num_classes, dtype=torch.long))

    def _reset_iou_components(self, stage):
        if stage == "train":
            self.train_intersections.zero_()
            self.train_unions.zero_()
        elif stage == "val":
            self.val_intersections.zero_()
            self.val_unions.zero_()
        elif stage == "test":
            self.test_intersections.zero_()
            self.test_unions.zero_()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)
        ce_loss = F.cross_entropy(
            logits,
            masks,
            ignore_index=self.ignore_index,
        )

        dice_loss = DiceLoss(logits, masks, self.classes, self.ignore_index)
        # focal_loss = FocalLoss(
        #     logits=logits,
        #     targets=msks,
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=self.ignore_index,
        #     labels_smoothing=0.1,
        # )

        loss = 0.5 * ce_loss + 0.5 * dice_loss
        preds = logits.argmax(dim=1)
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )

        self.train_unions += union
        self.train_intersections += inter

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"])

        return loss

    def on_train_epoch_end(self):
        m_iou, _ = iou_calculation(self.train_intersections, self.train_unions)
        self.log("train/mIoU", m_iou, on_epoch=True)
        self._reset_iou_components("train")

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch

        # 첫 번째 이미지 / 마스크 저장
        if batch_idx == 0:
            self.first_val_batch = (imgs.cpu(), masks.cpu())

        logits = self(imgs)
        ce_loss = F.cross_entropy(
            logits,
            masks,
            ignore_index=self.ignore_index,
        )
        dice_loss = DiceLoss(logits, masks, self.classes, self.ignore_index)
        # focal_loss = FocalLoss(
        #     logits=logits,
        #     targets=msks,
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=self.ignore_index,
        #     labels_smoothing=0.1,
        # )

        loss = 0.5 * ce_loss + 0.5 * dice_loss

        preds = torch.argmax(logits, dim=1)
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )
        self.val_intersections += inter.to(self.device)
        self.val_unions += union.to(self.device)

        self.validation_step_outputs.append(
            {
                "loss": loss.detach(),
            }
        )

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val/loss", avg_loss, on_epoch=True)

        m_iou, _ = iou_calculation(self.val_intersections, self.val_unions)

        self.log("val/mIoU", m_iou, on_epoch=True)

        if self.first_val_batch:
            first_img_tensor, first_mask_tensor = self.first_val_batch

            self.model.eval()
            with torch.no_grad():
                logits = self.model(first_img_tensor[0:1].to(self.device))
                preds = torch.argmax(logits, dim=1).cpu()

            self.model.train()

            gt_mask_color = apply_color_map(first_mask_tensor[0], self.classes)
            pred_mask_color = apply_color_map(preds[0], self.classes)

            w, h = gt_mask_color.size
            combined = Image.new("RGB", (w * 2, h))

            combined.paste(gt_mask_color, (0, 0))
            combined.paste(pred_mask_color, (w, 0))

            self.logger.experiment.log(
                {
                    "val/visualize": wandb.Image(
                        combined,
                        caption=f"Epoch {self.current_epoch}: Ground Truth | Prediction",
                    ),
                }
            )

            self.first_val_batch = None
        self._reset_iou_components("val")

    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        logits = self(imgs)

        ce_loss = F.cross_entropy(
            logits,
            masks,
            ignore_index=self.ignore_index,
        )
        dice_loss = DiceLoss(logits, masks, self.classes, self.ignore_index)
        # focal_loss = FocalLoss(
        #     logits=logits,
        #     targets=msks,
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=self.ignore_index,
        #     labels_smoothing=0.1,
        # )
        # )
        loss = 0.5 * ce_loss + 0.5 * dice_loss

        preds = logits.argmax(dim=1)

        self.test_step_outputs.append({"loss": loss.detach()})
        inter, union = iou_component(
            preds, masks, self.classes, ignore_idx=self.ignore_index
        )

        self.test_intersections += inter
        self.test_unions += union
        return loss

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()

        self.log("test/loss", avg_loss, on_epoch=True)
        self._reset_iou_components("test")

    def configure_optimizers(self):
        encoder_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            # Append encoder parameters for updating with lower learning rate
            if any(
                x in name
                for x in [
                    "encoder.stem",
                    "encoder.stage0",
                    "encoder.stage1",
                    "encoder.stage2",
                    "encoder.stage3",
                    "encoder.stage4",
                    "encoder.final_conv",
                ]
            ):
                encoder_params.append((name, param))
            # Append encoder parameters that are frozen
            # elif any(
            #     x in name
            #     for x in [
            #         "encoder.stage4",
            #         "encoder.final_conv",
            #     ]
            # ):
            # elif:
            #     continue

            else:
                decoder_params.append((name, param))

        def split_wd(param_list, base_lr):
            wd, nwd = [], []
            for name, p in param_list:
                if p.ndim <= 1 or "bias" in name or "norm" in name.lower():
                    nwd.append(p)
                else:
                    wd.append(p)

            return [
                {"params": wd, "weight_decay": 1e-4, "lr": base_lr},
                {"params": nwd, "weight_decay": 0.0, "lr": base_lr},
            ]

        shallow_lr = 1e-04
        optim_gropups = []
        optim_gropups += split_wd(encoder_params, shallow_lr)
        optim_gropups += split_wd(decoder_params, self.learning_rate)

        optimizer = torch.optim.AdamW(optim_gropups, betas=(0.9, 0.999))
        scheduler = CosineAnnealingWithWarmupLR(
            optimizer, T_max=200, eta_min=1e-6, warmup_epochs=10
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }


if __name__ == "__main__":
    from torchinfo import summary

    model = lt_MeTU(
        model_size="xxs", encoder_pretrained=True, learning_rate=1e-3, classes=19
    )
    summary(model, (1, 3, 512, 1024))
