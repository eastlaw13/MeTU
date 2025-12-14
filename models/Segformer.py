import os
import sys
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torchinfo import summary
from transformers import SegformerForSemanticSegmentation, logging

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.iou import iou_component, iou_calculation
from utils.Loss import DiceLoss, FocalLoss
from utils.lr_schedule import CosineAnnealingWithWarmupLR


class SegFormerb0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=self.num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs.logits

        return logits


class lt_segformerb0(L.LightningModule):
    def __init__(
        self, learning_rate: float, num_classes: int = 19, ignore_index: int = 255
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self._init_iou_components()

        self.model = SegFormerb0(num_classes=self.num_classes)

    def _init_iou_components(self):
        num_classes = self.num_classes
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
        images, masks = batch
        logits = self(images)

        # Interpolating
        logits = F.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        # Calculate the loss
        ce_loss = F.cross_entropy(logits, masks, ignore_index=self.ignore_index)
        dice_loss = DiceLoss(logits, masks, 19, 255)
        # focal_loss = FocalLoss(
        #     logits=logits,
        #     targets=msks,
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=self.ignore_index,
        #     labels_smoothing=0.1,
        # )
        loss = 0.5 * ce_loss + 0.5 * dice_loss

        # Calculate the metric
        preds = logits.argmax(dim=1)
        inter, union = iou_component(
            preds, masks, self.num_classes, ignore_idx=self.ignore_index
        )

        self.train_intersections += inter
        self.train_unions += union

        # Logging
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"])

        return loss

    def on_train_epoch_end(self):
        mIoU, _ = iou_calculation(self.train_intersections, self.train_unions)
        self.log("train/mIoU", mIoU, on_epoch=True)
        self._reset_iou_components("train")

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        # Interpolating
        logits = F.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        # Calculate the loss
        ce_loss = F.cross_entropy(logits, masks, ignore_index=self.ignore_index)
        dice_loss = DiceLoss(logits, masks, 19, self.ignore_index)
        # focal_loss = FocalLoss(
        #     logits=logits,
        #     targets=msks,
        #     alpha=0.25,
        #     gamma=2.0,
        #     ignore_index=self.ignore_index,
        #     labels_smoothing=0.1,
        # )

        loss = 0.5 * ce_loss + 0.5 * dice_loss

        # Calculate the metric
        preds = logits.argmax(dim=1)
        inter, union = iou_component(
            preds, masks, self.num_classes, ignore_idx=self.ignore_index
        )

        self.val_intersections += inter.to(self.device)
        self.val_unions += union.to(self.device)

        self.validation_step_outputs.append({"loss": loss.detach()})

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        self.log("val/loss", avg_loss, on_epoch=True)

        mIoU, _ = iou_calculation(self.val_intersections, self.val_unions)
        self.log("val/mIoU", mIoU, on_epoch=True)
        self._reset_iou_components("val")

    def test_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)

        # Interpolating
        logits = F.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        ce_loss = F.cross_entropy(logits, masks, ignore_index=self.ignore_index)
        dice_loss = DiceLoss(logits, masks, 19, self.ignore_index)
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

        self.test_step_outputs.append({"loss": loss.detach()})
        inter, union = iou_component(
            preds, masks, self.num_classes, ignore_idx=self.ignore_index
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
            if "encoder" in name:
                encoder_params.append((name, param))
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
    model = lt_segformerb0(learning_rate=5e-04)

    summary(model, (1, 3, 512, 1024))
