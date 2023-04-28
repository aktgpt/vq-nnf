import albumentations as aug
import colorcet as cc
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from torch import nn
from umap import UMAP

from models.resnet import ResNetHyperColumn
from models.efficientnet import EfficientNetHyperColumn
from kornia.filters import gaussian_blur2d


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device):
        self.device = device

    def __call__(self, image):
        image_copy = image.transpose((2, 0, 1))
        image_torch = torch.from_numpy(image_copy).float().to(self.device)
        return image_torch


class PixelFeatureExtractor:
    def __init__(self, model_name, num_features, device="0"):

        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.num_features = num_features

        if self.num_features != 27:
            if "resnet" in model_name:
                self.model = ResNetHyperColumn(model_name, 3, num_features).to(self.device)
            elif "efficientnet" in model_name:
                self.model = EfficientNetHyperColumn(model_name, 3, num_features).to(self.device)

        self.transform = ToTensor(self.device)
        self.augment = aug.Compose([aug.Normalize(p=1)])

    def get_color_features(self, image):
        # get color features in 3x3 neighborhood as vector of each pixel in image
        with torch.no_grad():
            # augmented = self.augment(image=image)
            # image_norm = augmented["image"]
            image_torch = self.transform(image) / 255
            image_feature = torch.cat(
                [
                    image_torch,
                    torch.roll(image_torch, shifts=[0, 1], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[0, -1], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[1, 0], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[-1, 0], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[1, 1], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[-1, -1], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[1, -1], dims=[1, 2]),
                    torch.roll(image_torch, shifts=[-1, 1], dims=[1, 2]),
                ],
                dim=0,
            )
            # image_feature = (image_feature - image_feature.mean(dim=(1, 2), keepdim=True)) / (
            #     image_feature.std(dim=(1, 2), keepdim=True) + 1e-5
            # )
        return image_feature

    def get_features(self, image):
        with torch.no_grad():
            if self.num_features == 27:
                image_feature = self.get_color_features(image)
            else:
                self.model.eval()
                # image_torch = self.transform(image) #/ 255
                augmented = self.augment(image=image)
                image_norm = augmented["image"]
                image_norm_torch = self.transform(image_norm)
                image_feature = self.model(image_norm_torch.unsqueeze(0)).squeeze(0)
                # image_feature = torch.concat([image_feature, image_torch], dim=0)
                # image_feature = gaussian_blur2d(image_norm_torch.unsqueeze(0), (3, 3), (1.5, 1.5)).squeeze(0)

        return image_feature

