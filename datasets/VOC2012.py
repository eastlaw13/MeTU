import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from typing import Union
from pathlib import Path
from torch.utils.data import Dataset

cfg_path = Path("../../data/VOC2012/eda/cfg/meta_info.json")
weights_pt_path = Path("../../data/VOC2012/eda/cfg/train_sampler_weights.pt")

try:
    with cfg_path.open(mode="r", encoding="utf-8") as f:
        CFG = json.load(f)
except FileNotFoundError:
    print(f"Error: File not found. Please check the path: {cfg_path}")
except json.JSONDecodeError:
    print(f"Error: The format of the JSON file is invalid: {cfg_path}")
except Exception as e:
    print(f"An error occurred while processing the file: {e}")


class VOC2012(Dataset):
    """
    PASCAL VOC 2012 Dataset.

    Args:
        mode(str): "train", "val", "test"
        transforms
    """

    def __init__(self, mode: str = "train", transforms=None):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.image_dir = Path("../../data/VOC2012/images") / self.mode
        self.mask_dir = Path("../../data/VOC2012/masks") / self.mode
        self.image_path = sorted(list(self.image_dir.rglob("*.jpg")))
        self.mask_path = sorted(list(self.mask_dir.rglob("*.png")))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = Image.open(self.image_path[index]).convert("RGB")
        mask = Image.open(self.mask_path[index]).convert("L")

        if self.transforms:
            image, mask = self.transforms(image, mask)

        return image, mask


class Traintransforms:
    def __init__(
        self,
        size=(512, 512),
        hflip_prob=0.5,
        color_jitter_prob=0.5,
        gamma_prob=0.3,
        blur_prob=0.3,
        noise_prob=0.2,
        auto_contrast_prob=0.3,
    ):
        self.size = size
        self.hflip_prob = hflip_prob
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob
        self.gamma_prob = gamma_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.auto_contrast_prob = auto_contrast_prob
        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        self.gausian_blur = T.GaussianBlur(kernel_size=(5), sigma=(0.1, 2.0))

        self.image_interp = T.InterpolationMode.BILINEAR
        self.mask_interp = T.InterpolationMode.NEAREST

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = F.resize(image, self.size, interpolation=self.image_interp)
        mask = F.resize(mask, self.size, interpolation=self.mask_interp)

        if torch.rand(1) < self.hflip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if torch.rand(1) < self.color_jitter_prob:
            image = self.color_jitter(image)

        if torch.rand(1) < self.gamma_prob:
            gamma = torch.empty(1).uniform_(0.5, 2.0).item()
            image = F.adjust_gamma(image, gamma)

        if torch.rand(1) < self.auto_contrast_prob:
            image = F.autocontrast(image)

        blur_noise_rand = torch.rand(1)

        if blur_noise_rand < self.blur_prob:
            image = self.gausian_blur(image)

        elif blur_noise_rand < self.blur_prob + self.noise_prob:
            image_tensor = F.to_tensor(image)
            noise = torch.rand_like(image_tensor) * 0.05
            image_tensor = torch.clamp(image_tensor + noise, 0.0, 1.0)
            image = F.to_pil_image(image_tensor)

        if isinstance(image, Image.Image):
            image = F.to_tensor(image)

        image = F.normalize(image, mean=CFG["train"]["mean"], std=CFG["train"]["std"])
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


class Valtransforms:
    def __init__(self, size=(512, 512)):
        self.size = size
        self.image_interp = F.InterpolationMode.BILINEAR
        self.mask_interp = F.InterpolationMode.NEAREST

    def __call__(self, image, mask):
        image = F.resize(image, self.size, interpolation=self.image_interp)
        mask = F.resize(mask, self.size, interpolation=self.mask_interp)

        image = F.to_tensor(image)
        image = F.normalize(image, mean=CFG["train"]["mean"], std=CFG["train"]["std"])
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


def load_weight_sampler(
    weights_path: Union[str, Path] = weights_pt_path,
    num_samples: Union[int, None] = None,
    replacement: bool = True,
) -> torch.utils.data.WeightedRandomSampler:
    """
    Load pre-calculated Train weight sampler from .pt file

    Args:
        weights_path (Union[str, Path]): 저장된 가중치 .pt 파일의 경로입니다.
        num_samples (Union[int, None]): 샘플링할 총 횟수.
                                        None이면 (기본값) 가중치의 개수(즉, 데이터셋 크기)와 같습니다.
        replacement (bool): 가중치에 따라 복원 추출(True)할지 비복원 추출(False)할지 결정합니다.

    Returns:
        torch.utils.data.WeightedRandomSampler: 생성된 가중치 랜덤 샘플러 객체.

    Raises:
        FileNotFoundError: 파일을 찾을 수 없을 때 발생합니다.
        Exception: PyTorch 로딩 중 오류가 발생할 때 발생합니다.

    """
    try:
        # 1. torch.load()를 사용하여 .pt 파일에서 PyTorch Tensor를 로드합니다.
        weights_tensor = torch.load(weights_path)

        # 텐서가 1차원인지 확인
        if weights_tensor.dim() != 1:
            raise ValueError(
                f"Loaded weights tensor must be 1D, but got {weights_tensor.dim()}D. Shape: {weights_tensor.shape}"
            )

        # num_samples를 지정하지 않았으면, 가중치의 개수를 사용합니다.
        if num_samples is None:
            num_samples = len(weights_tensor)

        # 2. WeightedRandomSampler를 생성합니다.
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights_tensor, num_samples=num_samples, replacement=replacement
        )

        print(f"WeightedRandomSampler successfully created from: {weights_path}")
        print(f"Sampler Size (num_samples): {num_samples}")
        return sampler

    except FileNotFoundError:
        print(f"Error: Weights file not found at: {weights_path}")
        raise
    except Exception as e:
        print(f"An error occurred while loading or creating sampler: {e}")
        raise
