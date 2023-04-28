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

from kmeans_pytorch import bisecting_kmeans, kmeans
from matchers.weight_calculator import WeightCalculator

haar_weight_dict = {
    "haar_1x": {
        "weights": torch.tensor([[1, -1], [-1, 1]]).float(),
        "dilation_x": 1,
        "dilation_y": 1,
        "filter_weight": 1,
    },
    "haar_2x": {
        "weights": torch.tensor([[1, -2, 1], [-1, 2, -1]]).float(),
        "dilation_x": 1,
        "dilation_y": 2,
        "filter_weight": 0.25,
    },
    "haar_2y": {
        "weights": torch.tensor([[1, -1], [-2, 2], [1, -1]]).float(),
        "dilation_x": 2,
        "dilation_y": 1,
        "filter_weight": 0.25,
    },
    "haar_3x": {
        "weights": torch.tensor([[-1, 2, -2, 1], [1, -2, 2, -1]]).float(),
        "dilation_x": 1,
        "dilation_y": 3,
        "filter_weight": 0.25,
    },
    "haar_3y": {
        "weights": torch.tensor([[-1, 1], [2, -2], [-2, 2], [1, -1]]).float(),
        "dilation_x": 3,
        "dilation_y": 1,
        "filter_weight": 0.25,
    },
    "haar_4xy": {
        "weights": torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]).float(),
        "dilation_x": 2,
        "dilation_y": 2,
        "filter_weight": 0.25,
    },
}

# "haar_4xy": {
#     "weights": torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]).float(),
#     "dilation_x": 2,
#     "dilation_y": 2,
#     "filter_weight": 0.2,
# },


def get_center_crop_coords(width: int, height: int, crop_width: int, crop_height: int):
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    y1 = (height - crop_height) // 2
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
            self.get_haar_filters(n_channels, kernel_size, scale_val, scale_weight)

        self.weights = weights
        self.template_features = []
        self.eps = 1e-8
        self.thresh = nn.Threshold(-1, -1)

    def get_haar_filters(self, n_channels, kernel_size, scale, scale_weight):
        w = int(kernel_size[0] * scale)
        h = int(kernel_size[1] * scale)
        for filter_name, filter_dict in islice(haar_weight_dict.items(), 0, self.n_haar_features):
            weights = filter_dict["weights"]
            w_k, h_k = weights.shape
            dil_x = filter_dict["dilation_x"]
            dil_y = filter_dict["dilation_y"]
            filter = nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=(w_k, h_k),
                dilation=(w // dil_x, h // dil_y),
                bias=False,
                groups=n_channels,
            )
            filter.weight.data = weights[None, None].repeat(n_channels, 1, 1, 1) / (w * h)
            self.kernel_sizes.append((w, h))
            self.filters.append(filter.to(self.device))
            filter_weight = scale_weight
            if self.params_weights is not None:
                filter_weight *= self.params_weights[
                    int(re.search(r"\d+", filter_name).group(0)) - 1
                ]
                # if "_1" in filter_name:
                #     filter_weight *= self.params_weights[0]
                # if "_2" in filter_name:
                #     filter_weight *= self.params_weights[1]
                # elif "_3" in filter_name:
                #     filter_weight *= self.params_weights[2]
                # elif "_4" in filter_name:
                #     filter_weight *= self.params_weights[3]
            else:
                filter_weight *= filter_dict["filter_weight"]
            self.filter_weights.append(filter_weight)
            self.scales.append(scale)
            self.cutoff.append(1 / scale)  # * int(re.search(r"\d+", filter_name).group(0))

        # print("Filter weights: ", self.filter_weights)



    def get_template_features(self, template):
        for filter, kernel_size in zip(self.filters, self.kernel_sizes):
            y = self.forward_filter(template, filter, kernel_size, (1, 1))
            self.template_features.append(y)
            # print(y.abs().max().item(), y.abs().min().item())

    def get_qeury_map(self, x):
        out_shape = None
        for i, (filter, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            y = self.forward_filter(x, filter, kernel_size, out_shape)
            if i == 0:
                out_shape = y.shape[2:]
                distances = torch.zeros(y.shape[2:], device=self.device)

            # calculate score by making a gaussian around the template feature and the std of the gaussian is the template feature
            distance = torch.exp(
                -0.25
                * (torch.abs(y - self.template_features[i]))
                # / ((self.template_features[i].abs() + self.eps))  # ** 2
            )
            distance = distance.mean(dim=1).squeeze(0) * self.filter_weights[i]
            # print(distance.max().item(), distance.min().item())

            # # get cosine similarity
            # distance = (
            #     F.cosine_similarity(y, self.template_features[i], dim=1, eps=self.eps)
            #     .squeeze(0)
            #     .abs()
            # )
            # distance = distance * self.filter_weights[i]

            # # get euclidean distance
            # distance = torch.abs(y - self.template_features[i]) * self.weights[None, :, None, None]
            # ##print(distance.max().item(), distance.min().item())
            # # # distance = ((y - self.template_features[i]) ** 2) * self.weights[None, :, None, None]
            # distance = -distance.sum(dim=1).squeeze(0) * self.filter_weights[i]

            # plt.figure()
            # plt.imshow(distance.cpu().numpy())
            distances += distance

        # plt.figure()
        # plt.imshow(distances.cpu().numpy())
        # plt.show()
        return distances

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

