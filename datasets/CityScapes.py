import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class CityScapes(Dataset):
    """
    CityScapes Datasets.

    Args:
        mode(str): "train", "val", "test"
        transforms: Argumentation and transforms
    """

    def __init__(self, mode, transforms=None):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.image_dir = Path("../../data/CityScapes/images") / self.mode
        self.mask_dir = Path("../../data/CityScapes/masks") / self.mode
        self.image_path = list(self.image_dir.rglob("*.png"))
        self.mask_path = list(self.mask_dir.rglob("*.png"))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img = Image.open(self.image_path[idx]).convert("RGB")
        msk = Image.open(self.mask_path[idx])

        if self.transforms:
            img, msk = self.transforms(img, msk)

        return img, msk


class NewTrainTransforms:
    def __init__(
        self,
        crop_size=(512, 1024),
        hflip_prob=0.5,
        color_jitter_prob=0.5,
        gamma_prob=0.3,
        blur_prob=0.2,
        noise_prob=0.2,
        autocontrast_prob=0.3,
    ):

        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.color_jitter_prob = color_jitter_prob
        self.gamma_prob = gamma_prob
        self.blur_prob = blur_prob
        self.noise_prob = noise_prob
        self.autocontrast_prob = autocontrast_prob
        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )
        self.gausian_blur = T.GaussianBlur(kernel_size=(5), sigma=(0.1, 2.0))

    def __call__(self, image, mask):
        i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        if torch.rand(1) < self.hflip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        if torch.rand(1) < self.color_jitter_prob:
            image = self.color_jitter(image)

        if torch.rand(1) < self.gamma_prob:
            gamma = torch.empty(1).uniform_(0.7, 1.3).item()
            image = F.adjust_gamma(image, gamma)

        if torch.rand(1) < self.autocontrast_prob:
            image = F.autocontrast(image)

        blur_noise_rand = torch.rand(1)

        if blur_noise_rand < self.blur_prob:
            image = self.gausian_blur(image)
        elif blur_noise_rand < self.blur_prob + self.noise_prob:
            img_tensor = F.to_tensor(image)
            noise = torch.randn_like(img_tensor) * 0.05
            img_tensor = torch.clamp(img_tensor + noise, 0.0, 1.0)
            image = F.to_pil_image(img_tensor)

        image = F.to_tensor(image)
        image = F.normalize(
            image, mean=[0.288, 0.326, 0.285], std=[0.186, 0.189, 0.186]
        )
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


class TrainTransforms:
    def __init__(self, crop_size=(512, 1024), hflip_prob=0.5):
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

    def __call__(self, image, mask):
        i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        if torch.rand(1) < self.hflip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        image = self.color_jitter(image)

        image = F.to_tensor(image)
        image = F.normalize(
            image, mean=[0.287, 0.325, 0.284], std=[0.176, 0.181, 0.178]
        )
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


class ValTransforms:
    def __init__(self, crop_size=(512, 1024)):
        self.crop_size = crop_size

    def __call__(self, image, mask):
        image = F.center_crop(image, self.crop_size)
        mask = F.center_crop(mask, self.crop_size)

        image = F.to_tensor(image)
        image = F.normalize(
            image, mean=[0.288, 0.326, 0.285], std=[0.186, 0.189, 0.186]
        )
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


if __name__ == "__main__":
    ts_train = NewTrainTransforms()
    ds_train = CityScapes(mode="train", transforms=ts_train)

    image, mask = ds_train[0]
    if ts_train is not None:
        image = torch.permute(image, (1, 2, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()
