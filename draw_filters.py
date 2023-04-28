import math
import re
from itertools import islice

import colorcet as cc
import ddks
import kornia
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from kornia.morphology import dilation, erosion, opening
from scipy import ndimage
from skimage.color import label2rgb
from torch import nn
from skimage.filters import gabor_kernel
from kmeans_pytorch import bisecting_kmeans, kmeans
import cv2
import decimal
import time
from fast_pytorch_kmeans import KMeans
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

haar_weight_dict = {
    "haar_1x": {
        "weights": torch.tensor([[1]]).float(),
        "dilation_x": 1,
        "dilation_y": 1,
        "filter_weight": 1,
    },
    "haar_2x": {
        "weights": torch.tensor([[1, -1]]).float(),
        "dilation_x": 1,
        "dilation_y": 2,
        "filter_weight": 0.25,
    },
    "haar_2y": {
        "weights": torch.tensor([[1], [-1]]).float(),
        "dilation_x": 2,
        "dilation_y": 1,
        "filter_weight": 0.25,
    },
    "haar_3x": {
        "weights": torch.tensor([[-1, 2, -1]]).float(),
        "dilation_x": 1,
        "dilation_y": 3,
        "filter_weight": 0.25,
    },
    "haar_3y": {
        "weights": torch.tensor([[-1], [2], [-1]]).float(),
        "dilation_x": 3,
        "dilation_y": 1,
        "filter_weight": 0.25,
    },
    # "haar_4xy": {
    #     "weights": torch.tensor([[1, -1], [-1, 1]]).float(),
    #     "dilation_x": 2,
    #     "dilation_y": 2,
    #     "filter_weight": 1.00,
    # },
}
cmaps = {}

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(category, cmap_list):
    # Create figure and adjust figure height to number of colormaps
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99)
    axs[0].set_title(f"{category} colormaps", fontsize=14)

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect="auto", cmap=name)
        ax.text(-0.01, 0.5, name, va="center", ha="right", fontsize=10, transform=ax.transAxes)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()

    # Save colormap list for later.
    cmaps[category] = cmap_list


plot_color_gradients("Perceptually Uniform Sequential", ["viridis"])
plt.savefig("exps_final/figures/filters_vis/color_gradients.pdf", dpi=300, bbox_inches="tight")
plt.close()


def convert_box_to_integral(box_filter):
    haar_multiplier = torch.tensor([[1, -1], [-1, 1]]).float()
    integral_filter = torch.zeros(box_filter.shape[0] + 1, box_filter.shape[1] + 1)
    for i in range(box_filter.shape[0]):
        for j in range(box_filter.shape[1]):
            integral_filter[i : i + 2, j : j + 2] += box_filter[i, j] * haar_multiplier
    return integral_filter


def get_gaussian_box_filter(kernel_size, sigma, order=(0, 0)):
    kernel = np.zeros(kernel_size)
    center_x = kernel_size[0] // 2
    center_y = kernel_size[1] // 2
    if kernel_size[0] % 2 == 0:
        x_ones = [center_x - 1, center_x + 1]
        x_mult = 0.5
    else:
        x_ones = [center_x, center_x + 1]
        x_mult = 1
    if kernel_size[1] % 2 == 0:
        y_ones = [center_y - 1, center_y + 1]
        y_mult = 0.5
    else:
        y_ones = [center_y, center_y + 1]
        y_mult = 1
    kernel[x_ones[0] : x_ones[1], y_ones[0] : y_ones[1]] = x_mult * y_mult
    kernel = ndimage.gaussian_filter(kernel, sigma=sigma, order=order)
    return kernel


def get_gauss_haar_integral(kernel_size, sigma, haar_filter_dict):
    haar_weight = haar_filter_dict["weights"]
    haar_weight_shape = haar_weight.shape
    new_kernel_shape_x = (
        kernel_size
        if kernel_size % haar_weight_shape[0] == 0
        else (kernel_size // haar_weight_shape[0] + 1) * haar_weight_shape[0]
    )
    new_kernel_shape_y = (
        kernel_size
        if kernel_size % haar_weight_shape[1] == 0
        else (kernel_size // haar_weight_shape[1] + 1) * haar_weight_shape[1]
    )

    gaussian_filter = get_gaussian_box_filter((new_kernel_shape_x, new_kernel_shape_y), sigma)
    repeat_x = new_kernel_shape_x // haar_weight_shape[0]
    repeat_y = new_kernel_shape_y // haar_weight_shape[1]
    haar_filter = haar_weight.repeat_interleave(repeat_x, axis=0).repeat_interleave(
        repeat_y, axis=1
    )
    haar_filter = haar_filter * gaussian_filter
    haar_integral_filter = convert_box_to_integral(haar_filter)

    return haar_filter, haar_integral_filter


kernel_size = (3, 3)
sigma = 1
# x = torch.ones(1, 1, 15, 15)
# x[:, :, 7, 7] = 1
# haar_filter = get_gauss_haar_integral(kernel_size[0], sigma, haar_weight_dict["haar_1x"])
# conv2d1 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=0, bias=False)
# conv2d1.weight.data = torch.tensor(haar_filter).float().unsqueeze(0).unsqueeze(0).clone()
# haar_filter = get_gauss_haar_integral(kernel_size[0], sigma, haar_weight_dict["haar_2x"])
# conv2d2 = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=0, bias=False)
# conv2d2.weight.data = torch.tensor(haar_filter).float().unsqueeze(0).unsqueeze(0).clone()
# y = conv2d1(x)
# y = conv2d2(y)


for haar_feat in haar_weight_dict:
    haar_filter_dict = haar_weight_dict[haar_feat]
    haar_filter, haar_integral_filter = get_gauss_haar_integral(
        kernel_size[0], sigma, haar_filter_dict
    )

    plt.imshow(haar_filter, cmap="jet")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(
        f"exps_final/figures/filters_vis/{haar_feat}_kernel_{kernel_size[0]}.pdf",
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()

    plt.imshow(haar_integral_filter, cmap="jet")
    plt.xticks([])
    plt.yticks([])
    plt.savefig(
        f"exps_final/figures/filters_vis/{haar_feat}_kernel_{kernel_size[0]}_integral.pdf",
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()
