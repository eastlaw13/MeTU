import os
import sys
import timm
import wandb
import torch
import warnings
import numpy as np
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchinfo import summary
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
        skips = []
        out = self.stem(x)
        out = self.stage0(out)
        out = self.stage1(out)
        out = self.stage2(out)
        low_level_feature = out
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.final_conv(out)
        high_level_feature = out
        skips = [low_level_feature, high_level_feature]

        return out, skips


class AtrousConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding, dilation, bias=False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()

        # 1x1 Conv
        modules = [AtrousConv(in_channels, out_channels, 1, padding=0, dilation=1)]

        # Atrous Convs (3x3, rates 6, 12, 18 or similar)
        for rate in atrous_rates:
            modules.append(
                AtrousConv(in_channels, out_channels, 3, padding=rate, dilation=rate)
            )

        # Image Pooling
        modules.append(self._make_image_pooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Final 1x1 Conv to combine features
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def _make_image_pooling(self, in_channels, out_channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        res = []
        size = x.shape[-2:]
        for conv in self.convs:
            if isinstance(conv, nn.Sequential):  # Image Pooling
                pool_out = conv(x)
                # Upsample back to the size of the input feature map
                res.append(
                    F.interpolate(
                        pool_out, size=size, mode="bilinear", align_corners=False
                    )
                )
            else:
                res.append(conv(x))

        x = torch.cat(res, dim=1)
        return self.project(x)


class MobileViT_DeeplabV3Plus(nn.Module):
    def __init__(
        self, model_size: str = "xxs", encoder_pretrained: bool = True, classes: int = 1
    ):
        super().__init__()
        self.encoder = MobileViTEncoder(model_size, encoder_pretrained)
        self.classes = classes

        self.stage2 = self.encoder.stage2
        self.stage5 = self.encoder.final_conv

        if model_size == "xxs":
            HIGH_LEVEL_CHANNELS = 320
            ASPP_CHANNELS = 128
            LOW_LEVEL_CHANNELS = 48
        elif model_size == "xs":
            HIGH_LEVEL_CHANNELS = 384
            ASPP_CHANNELS = 128
            LOW_LEVEL_CHANNELS = 64
        elif model_size == "s":
            HIGH_LEVEL_CHANNELS = 640
            ASPP_CHANNELS = 256
            LOW_LEVEL_CHANNELS = 96

        self.aspp = ASPP(HIGH_LEVEL_CHANNELS, ASPP_CHANNELS, atrous_rates=[6, 12, 18])
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(LOW_LEVEL_CHANNELS, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(ASPP_CHANNELS + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, self.classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        low_level_feat = None
        high_level_feat = None

        x, skips = self.encoder(x)
        low_level_feat, high_level_feat = skips

        x = self.aspp(high_level_feat)
        low_level_feat = self.low_level_conv(low_level_feat)

        x = F.interpolate(
            x, size=low_level_feat.shape[-2:], mode="bilinear", align_corners=False
        )

        x = torch.concat([x, low_level_feat], dim=1)

        x = self.decoder(x)

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x


class lt_mobilevit_dlv3_p(L.LightningModule):
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
        self.model = MobileViT_DeeplabV3Plus(
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
    model = lt_mobilevit_dlv3_p(
        model_size="xxs", encoder_pretrained=False, learning_rate=1e-3, classes=19
    )
    summary(model, (1, 3, 512, 1024))
