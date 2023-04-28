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
    "haar_4xy": {
        "weights": torch.tensor([[1, -1], [-1, 1]]).float(),
        "dilation_x": 2,
        "dilation_y": 2,
        "filter_weight": 0.25,
    },
}


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
    return torch.tensor(kernel).float()


def convert_box_to_integral(box_filter):
    haar_multiplier = torch.tensor([[1, -1], [-1, 1]]).float()
    integral_filter = torch.zeros(box_filter.shape[0] + 1, box_filter.shape[1] + 1)
    for i in range(box_filter.shape[0]):
        for j in range(box_filter.shape[1]):
            integral_filter[i : i + 2, j : j + 2] += box_filter[i, j] * haar_multiplier
    return integral_filter


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
    # print(haar_filter)
    haar_integral_filter = convert_box_to_integral(haar_filter)
    return haar_integral_filter, haar_filter_dict["filter_weight"]


# x = 1
# box_filter = get_average_box_filter(3)
# haar_filter = convert_box_to_haar(box_filter)
# a = get_gaussian_box_filter(3, 1)
# b = convert_box_to_haar(a)
# a1 = get_laplacian_box_filter(5, 1)
# b1 = convert_box_to_integral(a1)


def get_center_crop_coords(width: int, height: int, crop_width: int, crop_height: int):
    x1 = (width - crop_width) // 2 if width > crop_width else 0
    x2 = x1 + crop_width
    y1 = (height - crop_height) // 2 if height > crop_height else 0
    y2 = y1 + crop_height
    return x1, y1, x2, y2


class HaarFilters(nn.Module):
    def __init__(
        self,
        n_channels,
        kernel_size,
        n_haar_features=1,
        scales=1,
        weights=None,
        params_weights=None,
        device="cpu",
    ):
        super().__init__()
        self.filter_kernel_size = 3
        self.sigmas = [(2, 2), (2, 1), (1, 2)]
        # self.orders = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.order_weights = [1, 1, 1]
        self.device = device
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_haar_features = n_haar_features

        self.filters = []
        self.kernel_sizes = []
        self.filter_weights = []

        scale_vals = np.linspace(1, 1 / scales, scales)
        scale_weights = np.linspace(1, 1 / scales, scales)
        self.params_weights = params_weights

        self.scales = []
        self.cutoff = []
        for scale_val, scale_weight in zip(scale_vals, scale_weights):
            self.get_filters(n_channels, kernel_size, scale_val, scale_weight)

        self.weights = weights
        self.template_features = []
        self.eps = 1e-8
        self.thresh = nn.Threshold(-1, -1)

    def get_filters(self, n_channels, kernel_size, scale, scale_weight):
        w = int(kernel_size[0] * scale)
        h = int(kernel_size[1] * scale)

        for filter_name, filter_dict in islice(haar_weight_dict.items(), 0, self.n_haar_features):
            int_filter_weights, filter_weight = get_gauss_haar_integral(
                self.filter_kernel_size, self.sigmas[0], filter_dict
            )
            filter_kernel_size = int_filter_weights.shape
            dilation = (
                (w) // (filter_kernel_size[0] - 1),
                (h) // (filter_kernel_size[1] - 1),
            )
            # kernel_size = (self.filter_kernel_size + 1, self.filter_kernel_size + 1)

            filter = nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=filter_kernel_size,
                dilation=dilation,
                bias=False,
                groups=n_channels,
            )
            filter.weight.data = int_filter_weights[None, None].repeat(n_channels, 1, 1, 1) / (
                w * h
            )
            self.kernel_sizes.append((w, h))
            self.filters.append(filter.to(self.device))
            self.filter_weights.append(scale_weight * filter_weight)

    def get_filters2(self, n_channels, kernel_size, scale, scale_weight):
        w = int(kernel_size[0] * scale)
        h = int(kernel_size[1] * scale)

        for filter_name, filter_dict in islice(haar_weight_dict.items(), 0, self.n_haar_features):
            int_filter_weights, filter_weight = get_gauss_haar_integral(
                self.filter_kernel_size, self.sigmas[0], filter_dict
            )
            filter_kernel_size = int_filter_weights.shape
            dilation = (
                (w) // (filter_kernel_size[0] - 1),
                (h) // (filter_kernel_size[1] - 1),
            )
            # kernel_size = (self.filter_kernel_size + 1, self.filter_kernel_size + 1)

            filter = nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=filter_kernel_size,
                dilation=dilation,
                bias=False,
                groups=n_channels,
            )
            filter.weight.data = int_filter_weights[None, None].repeat(n_channels, 1, 1, 1) / (
                w * h
            )
            self.kernel_sizes.append((w, h))
            self.filters.append(filter.to(self.device))
            self.filter_weights.append(scale_weight * filter_weight)

    def forward_filter(self, x, filter, kernel_size, out_shape=None):
        with torch.no_grad():
            pad_x = max(0, kernel_size[0] - x.shape[2] + 2) // 2
            pad_y = max(0, kernel_size[1] - x.shape[3] + 2) // 2
            x_pad = F.pad(x, (pad_y, pad_y, pad_x, pad_x))
            y_pad = filter(x_pad)
            if out_shape is not None:
                x1, y1, x2, y2 = get_center_crop_coords(
                    y_pad.shape[2], y_pad.shape[3], out_shape[0], out_shape[1]
                )
                y = y_pad[:, :, x1:x2, y1:y2]
            else:
                y = y_pad
            return y

    def get_template_features(self, template):
        for filter, kernel_size in zip(self.filters, self.kernel_sizes):
            y = self.forward_filter(template, filter, kernel_size, (1, 1))
            self.template_features.append(y)
            # print(y.abs().max().item(), y.abs().min().item())

    def get_qeury_map(self, x):
        out_shape = None
        distances = torch.zeros(x.shape[2:], device=self.device)
        for i, (filter, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            y = self.forward_filter(x, filter, kernel_size, out_shape)
            ## check out_shape and set it to the minimum of all the out_shapes
            if out_shape is None:
                out_shape = y.shape[2:]
                distances = torch.zeros(y.shape[2:], device=self.device)
            else:
                if not y.shape[2:] == out_shape:
                    out_shape = y.shape[2:]
                    x1, y1, x2, y2 = get_center_crop_coords(
                        distances.shape[0], distances.shape[1], out_shape[0], out_shape[1]
                    )
                    distances = distances[x1:x2, y1:y2]

            # ## clip the distance
            # distance = torch.abs(y - self.template_features[i])
            # distance = (
            #     torch.clamp(
            #         distance,
            #         torch.zeros_like(self.weights[None, :, None, None]),
            #         self.weights[None, :, None, None] * 10,
            #     )
            #     * self.weights[None, :, None, None]
            # )
            # distance = -distance.sum(dim=1).squeeze(0) * self.filter_weights[i]

            ## get relative distance
            # distance = (
            #     torch.abs(y - self.template_features[i])
            #     / self.template_features[i].abs()
            #     * self.weights[None, :, None, None]
            # )
            # if i == 0:
            #     out_shape = y.shape[2:]
            #     distances = torch.zeros(y.shape[2:], device=self.device)

            # calculate score by making a gaussian around the template feature and the std of the gaussian is the template feature
            # distance = torch.exp(
            #     -(torch.abs(y - self.template_features[i]))
            #     # / ((self.template_features[i].abs() + self.eps))  # ** 2
            # )
            # distance = distance.mean(dim=1).squeeze(0) * self.filter_weights[i]
            # print(distance.max().item(), distance.min().item())

            # # get cosine similarity
            # distance = (
            #     F.cosine_similarity(y, self.template_features[i], dim=1, eps=self.eps)
            #     .squeeze(0)
            #     .abs()
            # )
            # distance = distance * self.filter_weights[i]

            # get euclidean distance
            distance = torch.abs(y - self.template_features[i]) * self.weights[None, :, None, None]
            ##print(distance.max().item(), distance.min().item())
            # # distance = ((y - self.template_features[i]) ** 2) * self.weights[None, :, None, None]
            distance = -distance.sum(dim=1).squeeze(0) * self.filter_weights[i]

            # plt.figure()
            # plt.imshow(-distance.cpu().numpy())
            distances += distance

        # plt.figure()
        # plt.imshow(distances.cpu().numpy())
        # plt.show()
        return distances

    def get_qeury_map2(self, x):
        out_shape = None
        for i, (filter, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            y = self.forward_filter(x, filter, kernel_size, out_shape)
            if i == 0:
                out_shape = y.shape[2:]
                distances = torch.ones(y.shape[2:], device=self.device)

            ## calculate score by making a gaussian around the template feature and the std of the gaussian is the template feature
            # distance = torch.exp(
            #     -(torch.abs(y - self.template_features[i]))
            #     # / ((self.template_features[i].abs() + self.eps))  # ** 2
            # )
            # distance = distance.mean(dim=1).squeeze(0) * self.filter_weights[i]
            # print(distance.max().item(), distance.min().item())

            # # get cosine similarity
            # distance = (
            #     F.cosine_similarity(y, self.template_features[i], dim=1, eps=self.eps)
            #     .squeeze(0)
            #     .abs()
            # )
            # distance = distance * self.filter_weights[i]

            # get euclidean distance
            distance = torch.abs(y - self.template_features[i]) * self.weights[None, :, None, None]
            ##print(distance.max().item(), distance.min().item())
            # # distance = ((y - self.template_features[i]) ** 2) * self.weights[None, :, None, None]
            distance = distance.mean(dim=1).squeeze(0) * self.filter_weights[i]

            # plt.figure()
            # plt.imshow(distance.cpu().numpy())
            distances *= distance

        # distances = distances.squeeze(0).mean(dim=0)
        # plt.figure()
        # plt.imshow(-distances.cpu().numpy())
        # plt.show()
        return -distances

        # distance = torch.abs(y - self.template_features[i])
        # # np.set_printoptions(precision=4, suppress=True)
        # # print(np.percentile(distance.cpu().numpy(), [50, 75, 90, 95, 99, 100]))
        # distance = (
        #     torch.clip(
        #         distance,
        #         min=torch.zeros_like(self.template_features[i]),
        #         max=torch.quantile(distance.flatten(2, 3), 0.99, dim=2)[:, :, None, None],
        #     )
        #     * self.weights[None, :, None, None]
        # )
        # distance = -distance.sum(dim=1).squeeze(0) * self.filter_weights[i]

        # relative_distance = -(
        #     torch.abs(
        #         torch.clip(
        #             self.template_features[i] - y,
        #             min=-(torch.ones(y.shape) / self.n_channels / self.scales[i] / 2).to(
        #                 self.device
        #             ),
        #             max=(torch.ones(y.shape) / self.n_channels / self.scales[i] / 2).to(
        #                 self.device
        #             ),
        #         )
        #     )
        #     # / (self.template_features[i].abs() + self.eps)
        # )
        # # relative_distance = 1 + self.thresh(relative_distance)
        # distance = relative_distance.mean(dim=1).squeeze(0) * self.filter_weights[i]

        # similar_points = F.relu(
        #     torch.clip(
        #         y,
        #         min=-torch.abs(self.template_features[i]),
        #         max=torch.abs(self.template_features[i]),
        #     )
        #     / (self.template_features[i] + self.eps)
        # )
        # distance = similar_points.mean(dim=1).squeeze(0) * self.filter_weights[i]


class HaarTemplateMatcher:
    def __init__(
        self,
        template,
        template_shape,
        n_clusters=16,
        n_haar_features=1,
        scales=1,
        template_image=None,
        params_weights=None,
        verbose=False,
    ):
        self.device = template.device
        self.kernel = torch.ones(3, 3).to(self.device)
        self.eps = 1e-6
        self.n_clusters = n_clusters
        self.n_chunks = 8000
        self.colors = sns.color_palette(cc.glasbey, self.n_clusters)

        self.pool1d = nn.MaxPool1d(kernel_size=n_clusters, return_indices=True)
        self.unpool1d = nn.MaxUnpool1d(kernel_size=n_clusters)

        c, self.t_w, self.t_h = template.shape

        # self.use_xy = True
        # if self.use_xy:
        #     # get x-y coordinates
        #     x = torch.arange(self.t_h).repeat(self.t_w, 1).float().to(self.device) / self.t_h
        #     y = torch.arange(self.t_w).repeat(self.t_h, 1).t().float().to(self.device) / self.t_w
        #     template = torch.cat([template, x[None, :, :], y[None, :, :]], dim=0)

        template_flatten = template.reshape(-1, self.t_w * self.t_h).transpose(1, 0)

        self.cluster_centers, choice_cluster, _ = kmeans(
            X=template_flatten, num_clusters=n_clusters
        )
        # self.cluster_centers = cluster_centers[:, :-2] if self.use_xy else cluster_centers

        one_hot = (
            torch.nn.functional.one_hot(
                choice_cluster.reshape(self.t_w, self.t_h), num_classes=n_clusters
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        ).float()
        # dist_transform = kornia.contrib.distance_transform(one_hot, kernel_size=3)
        self.verbose = verbose
        if self.verbose:
            self.template_labels = label2rgb(
                one_hot.argmax(dim=1).squeeze(0).cpu().numpy(), colors=self.colors
            )
        else:
            self.template_labels = None

        cumsum_onehot = one_hot.cumsum(dim=2).cumsum(dim=3)

        # if template_image is not None:
        #     temp_nnf_onehot, _ = self.get_nnf(template_image)
        #     temp_histogram = temp_nnf_onehot.sum(dim=(1, 2))
        #     temp_weights = temp_histogram.sum() / temp_histogram
        #     temp_weights = temp_weights / temp_weights.sum()
        # else:
        temp_weights = torch.ones(n_clusters, device=self.device) / n_clusters
        ## get weights for each cluster based on the distance from the center
        # temp_weights_mat = np.zeros((self.t_w, self.t_h))
        # temp_weights_mat[self.t_w // 2, self.t_h // 2] = 1
        # gauss_filtered = ndimage.gaussian_filter(
        #     temp_weights_mat, sigma=(self.t_w // 4, self.t_h // 4)
        # )
        # temp_weights = torch.from_numpy(gauss_filtered).to(self.device) / gauss_filtered.sum()
        # temp_weights = (temp_weights * one_hot).sum(dim=(2, 3)).squeeze(0)

        self.rot_inv_conv = HaarFilters(
            n_clusters,
            (template_shape[0], template_shape[1]),
            n_haar_features=n_haar_features,
            scales=scales,
            weights=temp_weights,
            params_weights=params_weights,
            device=self.device,
        )
        self.rot_inv_conv.get_template_features(cumsum_onehot)
        self.gaussian_filter = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(self.t_w // 3, self.t_h // 3),
            stride=1,
            padding="same",
            padding_mode="reflect",
            bias=False,
        )
        self.gaussian_filter.weight.data = (
            torch.ones((self.t_w // 3, self.t_h // 3))
            .to(self.device)
            .reshape(1, 1, self.t_w // 3, self.t_h // 3)
        ) / ((self.t_w // 3) * (self.t_h // 3))
        # self.average_filter.weight.data = (
        #     torch.ones((self.t_w // 3, self.t_h // 3))
        #     .to(self.device)
        #     .reshape(1, 1, self.t_w // 3, self.t_h // 3)
        # ) / ((self.t_w // 3) * (self.t_h // 3))

    def get_heatmap(self, x):
        with torch.no_grad():
            one_hots, _, labels = self.get_nnf(x)
            integral_onehot = one_hots.cumsum(dim=1).cumsum(dim=2)
            deform_diff = self.rot_inv_conv.get_qeury_map(integral_onehot.unsqueeze(0))
            # deform_diff = kornia.filters.gaussian_blur2d(
            #     deform_diff[None, None, :, :],
            #     (2 * (self.t_w // 6) + 1, 2 * (self.t_h // 6) + 1),
            #     sigma=(self.t_w, self.t_h),
            # )[0, 0]
            # deform_diff = self.average_filter(deform_diff[None, None, :, :])[0, 0]
        return deform_diff, self.template_labels, labels

    def get_nnf(self, x):
        d, w, h = x.shape
        chunks = torch.split(x.reshape(d, w * h).transpose(1, 0), self.n_chunks, dim=0)

        nnf_idxs = []
        sim_vals = []
        one_hots = []
        sim_one_hots = []
        for i, chunk in enumerate(chunks):
            dist = -torch.cdist(self.cluster_centers, chunk)
            sim, idx = self.pool1d(dist.transpose(0, 1))
            one_hot = torch.ones_like(sim)
            one_hots.append(self.unpool1d(one_hot, idx))
            sim_one_hots.append(self.unpool1d(sim, idx))
            sim_vals.append(sim)
            nnf_idxs.append(idx)

        nnf_idxs = torch.cat(nnf_idxs).transpose(0, 1).reshape(w, h)
        sim_vals = -torch.cat(sim_vals).transpose(0, 1).reshape(1, w, h)
        sim_one_hots = -torch.cat(sim_one_hots).transpose(0, 1).reshape(-1, w, h)
        one_hots = torch.cat(one_hots).transpose(0, 1).reshape(-1, w, h)
        labels = one_hots.argmax(dim=0).cpu().numpy()
        if self.verbose:
            return one_hots, nnf_idxs, label2rgb(labels, colors=self.colors)
        else:
            return one_hots, nnf_idxs, labels
