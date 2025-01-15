import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, images: np.ndarray) -> torch.Tensor:
        images = images.transpose((0, 3, 1, 2))
        return torch.from_numpy(images).float().div(255.0)


class Normalize(object):
    def __init__(self, mean: float, std: float):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return images.sub_(self.mean[None, :, None, None]).div_(
            self.std[None, :, None, None]
        )


class ColorJitter(object):
    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        self.image_transforms = []

        if self.brightness is not None:
            if type(self.brightness) == tuple or type(self.brightness) == list:
                brightness_transform = (
                    lambda image: transforms.functional.adjust_brightness(
                        image, np.random.uniform(self.brightness[0], self.brightness[1])
                    )
                )
            else:
                brightness_transform = (
                    lambda image: transforms.functional.adjust_brightness(
                        image, self.brightness
                    )
                )
            self.image_transforms.append(brightness_transform)

        if self.contrast is not None or self.contrast != 0.0:
            if type(self.contrast) == tuple or type(self.contrast) == list:
                contrast_transform = (
                    lambda image: transforms.functional.adjust_contrast(
                        image, np.random.uniform(self.contrast[0], self.contrast[1])
                    )
                )
            elif type(self.contrast) == float:
                contrast_transform = (
                    lambda image: transforms.functional.adjust_contrast(
                        image, self.contrast
                    )
                )
            self.image_transforms.append(contrast_transform)

        if self.saturation is not None or self.saturation != 0.0:
            if type(self.saturation) == tuple or type(self.saturation) == list:
                saturation_transform = (
                    lambda image: transforms.functional.adjust_saturation(
                        image, np.random.uniform(self.saturation[0], self.saturation[1])
                    )
                )
            else:
                saturation_transform = (
                    lambda image: transforms.functional.adjust_saturation(
                        image, self.saturation
                    )
                )
            self.image_transforms.append(saturation_transform)

        if self.hue is not None or self.hue != 0.0:
            if type(self.hue) == tuple or type(self.hue) == list:
                hue_transform = lambda image: transforms.functional.adjust_hue(
                    image, np.random.uniform(self.hue[0], self.hue[1])
                )
            else:
                hue_transform = lambda image: transforms.functional.adjust_hue(
                    image, self.hue
                )
            self.image_transforms.append(hue_transform)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        assert images.ndim == 4, "images must be 4-dimentional."
        assert (
            type(images) == np.ndarray
        ), "images must be numpy.ndarray. Got {}".format(type(images))

        random.shuffle(self.image_transforms)

        jitterd_images = []
        for image in images:
            pil_image = Image.fromarray(image)
            for image_transform in self.image_transforms:
                pil_image = image_transform(pil_image)
            jitterd_images.append(np.asarray(pil_image))

        return np.stack(jitterd_images, axis=0)


class RandomHorizontalFlip(object):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, images: np.ndarray) -> np.ndarray:
        assert images.ndim == 4, "images must be 4-dimentional."
        assert (
            type(images) == np.ndarray
        ), "images must be numpy.ndarray. Got {}".format(type(images))
        if np.random.rand() < self.p:
            images = np.stack(
                [
                    transforms.functional.hflip(Image.fromarray(image))
                    for image in images
                ],
                axis=0,
            )
        return images


class RandomRotation(object):
    def __init__(self, degrees: float = 10) -> None:
        self.degrees = degrees

    def __call__(self, images: np.ndarray) -> np.ndarray:
        assert images.ndim == 4, "images must be 4-dimentional."
        assert (
            type(images) == np.ndarray
        ), "images must be numpy.ndarray. Got {}".format(type(images))
        angle = np.random.uniform(-self.degrees, self.degrees)
        images = np.stack(
            [
                transforms.functional.rotate(Image.fromarray(image), angle)
                for image in images
            ],
            axis=0,
        )
        return images


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d tensor.
    Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """

    def __init__(self, kernel_size: int = 15, sigma: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrid = torch.meshgrid(torch.arange(kernel_size))[0].float()

        mean = (kernel_size - 1) / 2
        kernel = kernel / (sigma * math.sqrt(2 * math.pi))
        kernel = kernel * torch.exp(-(((meshgrid - mean) / sigma) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        # kernel = kernel / torch.max(kernel)

        self.kernel = kernel.view(1, 1, *kernel.size())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        _, c, _ = inputs.shape
        inputs = F.pad(
            inputs,
            pad=((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
            mode="reflect",
        )
        kernel = self.kernel.repeat(c, *[1] * (self.kernel.dim() - 1)).to(inputs.device)
        return F.conv1d(inputs, weight=kernel, groups=c)
