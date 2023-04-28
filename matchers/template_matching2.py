import math

import colorcet as cc
import ddks
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from kornia.morphology import dilation, erosion, opening
from scipy import ndimage
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
    "haar_4xy": {
        "weights": torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]).float(),
        "dilation_x": 2,
        "dilation_y": 2,
        "filter_weight": 0.25,
    },
    "haar_3x": {
        "weights": torch.tensor([[-1, 2, -2, 1], [1, -2, 2, -1]]).float(),
        "dilation_x": 1,
        "dilation_y": 3,
        "filter_weight": 0.2,
    },
    "haar_3y": {
        "weights": torch.tensor([[-1, 1], [2, -2], [-2, 2], [1, -1]]).float(),
        "dilation_x": 3,
        "dilation_y": 1,
        "filter_weight": 0.2,
    },
}


def get_center_crop_coords(width: int, height: int, crop_width: int, crop_height: int):
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    return x1, y1, x2, y2


class HaarFilters(nn.Module):
    def __init__(
        self, n_channels, kernel_size, n_haar_features=1, scales=1, weights=None, device="cpu"
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
        for scale_val in scale_vals:
            self.get_haar_filters(n_channels, kernel_size, scale_val)

        self.weights = weights
        self.template_features = []

    def get_haar_filters(self, n_channels, kernel_size, scale):
        w = int(kernel_size[0] * scale)
        h = int(kernel_size[1] * scale)
        for _, filter_dict in haar_weight_dict.items()[0 : self.n_haar_features]:
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
            self.filter_weights.append(filter_dict["filter_weight"] * scale)

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

    def get_query_distance(self, x):
        out_shape = None
        for i, (filter, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            y = self.forward_filter(x, filter, kernel_size, out_shape)
            if i == 0:
                out_shape = y.shape[2:]
                distances = torch.zeros(y.shape[2:], device=self.device)
            distance = torch.abs(y - self.template_features[i]) * self.weights[None, :, None, None]
            distance = distance.sum(dim=1).squeeze(0)
            # plt.figure()
            # plt.imshow(distance[0].cpu().numpy())
            distance *= self.filter_weights[i]
            distances += distance
        # plt.figure()
        # plt.imshow(distances[0].cpu().numpy())
        # plt.show()
        return distances


class HaarTemplateMatcher:
    def __init__(self, template, n_clusters=16, template_image=None):
        self.device = template.device
        self.kernel = torch.ones(3, 3).to(self.device)
        self.eps = 1e-6
        self.n_clusters = n_clusters
        self.n_chunks = 1000

        self.pool1d = nn.MaxPool1d(kernel_size=n_clusters, return_indices=True)
        self.unpool1d = nn.MaxUnpool1d(kernel_size=n_clusters)

        c, self.t_w, self.t_h = template.shape
        template_flatten = template.reshape(c, self.t_w * self.t_h).transpose(1, 0)

        self.cluster_centers, choice_cluster, _ = kmeans(
            X=template_flatten, num_clusters=n_clusters
        )

        one_hot = (
            torch.nn.functional.one_hot(
                choice_cluster.reshape(self.t_w, self.t_h), num_classes=n_clusters
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        ).float()
        cumsum_onehot = one_hot.cumsum(dim=2).cumsum(dim=3)

        if template_image is not None:
            temp_nnf_onehot, _ = self.get_nnf(template_image)
            temp_histogram = temp_nnf_onehot.sum(dim=(1, 2))
            temp_weights = temp_histogram.sum() / temp_histogram
            temp_weights = temp_weights / temp_weights.sum()
        else:
            temp_weights = torch.ones(n_clusters, device=self.device) / n_clusters

        self.rot_inv_conv = HaarFilters(
            n_clusters, (self.t_w, self.t_h), weights=temp_weights, device=self.device
        )
        self.rot_inv_conv.get_template_features(cumsum_onehot)

    def get_heatmap(self, x):
        with torch.no_grad():
            one_hots, _ = self.get_nnf(x)

            integral_onehot = one_hots.cumsum(dim=1).cumsum(dim=2)

            deform_diff = self.rot_inv_conv.get_query_distance(integral_onehot.unsqueeze(0))

        return deform_diff

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

        return one_hots, nnf_idxs


def get_center_crop_coords(width: int, height: int, crop_width: int, crop_height: int):
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    return x1, y1, x2, y2


def get_haar_filters(n_channels, kernel_size, scales=4):
    filters = []
    scale_vals = np.linspace(0, 1, scales)
    for scale_val in range(scale_vals):
        filter = nn.Conv2d(
            n_channels,
            n_channels,
            kernel_size=(3, 3),
            dilation=((kernel_size[0] // scale_val) // 2, (kernel_size[1] // scale_val) // 2),
            stride=1,
            bias=False,
            groups=1,
        )
        filter.weight.data = (
            torch.tensor([[[0, 1, -1, 0], [1, 1, -1, -1], [-1, 1, -1, 1], [0, -1, 1, 0]]])
            .repeat(n_channels, 1, 1, 1)
            .float()
        )
    return filters


class RotationInvariantConv(nn.Module):
    def __init__(self, n_channels, kernel_size, scales=4, device="cpu"):
        super(RotationInvariantConv, self).__init__()
        self.device = device
        self.filters, self.kernel_sizes = self.get_rotinv_filters(n_channels, kernel_size, scales)
        self.template_features = None
        self.filter_weights = torch.tensor(
            [1 / (scales + 1 - i) for i in range(scales)], device=device
        )
        # self.filter_weights = self.filter_weights / self.filter_weights.sum()

    def get_rotinv_filters(self, n_channels, kernel_size, scales=4):
        # avg_kernel = (kernel_size[0] + kernel_size[1]) // 2
        filters = []
        kernel_sizes = []
        scale_vals = np.linspace(0.2, 0.8, num=scales)
        for scale_val in scale_vals:
            kernel_w = int(kernel_size[0] * scale_val)
            kernel_h = int(kernel_size[1] * scale_val)
            filter = nn.Conv2d(
                n_channels,
                n_channels,
                kernel_size=(4, 4),
                dilation=(kernel_w // 3, kernel_h // 3),
                bias=False,
                groups=n_channels,
            ).to(self.device)
            filter.weight.data = (
                torch.tensor(
                    [[[0, 1, -1, 0], [1, -1, 1, -1], [-1, 1, -1, 1], [0, -1, 1, 0]]],
                    device=self.device,
                )
                .repeat(n_channels, 1, 1, 1)
                .float()
            ) / (kernel_w * kernel_h)
            filters.append(filter)
            kernel_sizes.append((kernel_w, kernel_h))
        return filters, kernel_sizes

    def forward(self, x, out_shape):
        outs = []
        for i, (filter, kernel_size) in enumerate(zip(self.filters, self.kernel_sizes)):
            pad_x = max(0, kernel_size[0] - x.shape[2] + 1) // 2
            pad_y = max(0, kernel_size[1] - x.shape[3] + 1) // 2
            x_pad = F.pad(x, (pad_y, pad_y, pad_x, pad_x))
            out_pad = filter(x_pad)
            x1, y1, x2, y2 = get_center_crop_coords(
                out_pad.shape[2], out_pad.shape[3], out_shape[0], out_shape[1]
            )
            out = out_pad[:, :, x1:x2, y1:y2]
            if self.template_features is not None:
                out = torch.abs(out - self.template_features[i])
                out = out.mean(1)  # * self.filter_weights[i]
            outs.append(out)
        if self.template_features is not None:
            return torch.cat(outs, 0).mean(0)  # .squeeze(0)
        else:
            return outs

    def get_template_features(self, template):
        template_features = self.forward(template, (1, 1))
        self.template_features = template_features


class HaarTemplateMatcher:
    def __init__(
        self, template, n_clusters=16, filter_distances=False,
    ):
        self.device = template.device
        self.kernel = torch.ones(3, 3).to(self.device)
        self.eps = 1e-6
        self.n_clusters = n_clusters
        self.n_chunks = 1000
        self.temp_dim_scale = 8
        self.hist_abs = True
        self.norm_hist_diff = False
        self.deform_levels = 6

        self.pool1d = nn.MaxPool1d(kernel_size=n_clusters, return_indices=True)
        self.unpool1d = nn.MaxUnpool1d(kernel_size=n_clusters)

        c, self.t_w, self.t_h = template.shape
        template_flatten = template.reshape(c, self.t_w * self.t_h).transpose(1, 0)

        self.sum_filter = nn.Conv2d(
            self.n_clusters,
            self.n_clusters,
            (2, 2),
            dilation=(self.t_w, self.t_h),
            bias=False,
            groups=self.n_clusters,
        ).to(self.device)
        self.sum_filter.weight.data = (
            torch.tensor([[1, -1], [-1, 1]], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        ).repeat(self.n_clusters, 1, 1, 1)

        self.cluster_centers, choice_cluster, dis_vals = kmeans(
            X=template_flatten, num_clusters=n_clusters
        )

        self.one_hot = (
            torch.nn.functional.one_hot(
                choice_cluster.reshape(self.t_w, self.t_h), num_classes=n_clusters
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        ).float()
        cumsum_onehot = self.one_hot.cumsum(dim=2).cumsum(dim=3)

        self.rot_inv_conv = RotationInvariantConv(
            n_clusters, (self.t_w, self.t_h), device=self.device
        )
        self.rot_inv_conv.get_template_features(cumsum_onehot)
        self.temp_area_hist_norm = cumsum_onehot[:, :, -1, -1] / (self.t_w * self.t_h)

        if filter_distances:
            max_dis_vals = [[] for _ in range(n_clusters)]
            for i in range(n_clusters):
                cluster_distances = dis_vals[torch.where(choice_cluster == i)[0], i]
                max_dis_vals[i] = cluster_distances.max().item()
            self.max_dis_vals = torch.tensor(max_dis_vals).to(self.device)
        else:
            self.max_dis_vals = torch.ones(n_clusters).to(self.device) * 1e10

    def get_heatmap(self, x):
        with torch.no_grad():
            one_hots, _ = self.get_nnf(x)

            integral_onehot = one_hots.cumsum(dim=1).cumsum(dim=2)

            query_area_histogram = self.sum_filter(integral_onehot.unsqueeze(0)).squeeze(0).permute(
                1, 2, 0
            ) / (self.t_w * self.t_h)
            orig_shape = query_area_histogram.shape[:2]

            overall_hist_diff = torch.abs(query_area_histogram - self.temp_area_hist_norm).mean(-1)

            deform_diff = self.rot_inv_conv(integral_onehot.unsqueeze(0), orig_shape)

            # all_hist_diffs = torch.cat(
            #     [overall_hist_diff.mean(-1).unsqueeze(-1), deform_diff], dim=-1
            # )

            histogram_distance = overall_hist_diff + 1.0 * deform_diff

        return histogram_distance

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
        # Filter the distance map
        one_hots[sim_one_hots > self.max_dis_vals.unsqueeze(-1).unsqueeze(-1)] = 0
        # print("Removed pixels: ", ((w * h) - one_hots.sum()).item())
        return one_hots, nnf_idxs  # adjacency_distance


class DeformableTemplateMatcher:
    def __init__(
        self, template, n_clusters=16, filter_distances=False,
    ):
        self.device = template.device
        self.kernel = torch.ones(3, 3).to(self.device)
        self.eps = 1e-6
        self.n_clusters = n_clusters
        self.n_chunks = 1000
        self.temp_dim_scale = 8
        self.hist_abs = True
        self.norm_hist_diff = False
        self.deform_levels = 6

        self.pool1d = nn.MaxPool1d(kernel_size=n_clusters, return_indices=True)
        self.unpool1d = nn.MaxUnpool1d(kernel_size=n_clusters)

        c, self.t_w, self.t_h = template.shape
        template_flatten = template.reshape(c, self.t_w * self.t_h).transpose(1, 0)

        self.sum_filter = nn.Conv2d(
            self.n_clusters,
            self.n_clusters,
            (2, 2),
            dilation=(self.t_w, self.t_h),
            bias=False,
            groups=self.n_clusters,
        ).to(self.device)
        self.sum_filter.weight.data = (
            torch.tensor([[1, -1], [-1, 1]], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        ).repeat(self.n_clusters, 1, 1, 1)

        self.cluster_centers, choice_cluster, dis_vals = kmeans(
            X=template_flatten, num_clusters=n_clusters
        )
        _, template_histogram = torch.unique(choice_cluster, return_counts=True)
        self.temp_area_hist_norm = template_histogram / template_histogram.sum()

        self.one_hot = torch.nn.functional.one_hot(
            choice_cluster.reshape(self.t_w, self.t_h), num_classes=n_clusters
        )
        self.deform_filters = []
        self.deform_hist_norms = []
        self.deform_weights = torch.ones((self.deform_levels + 1, 1), device=self.device)
        self.deform_shapes = []
        for i in range(self.deform_levels):
            window_w = int(self.t_w * ((i + 1) / (self.deform_levels + 1)))
            window_h = int(self.t_h * ((i + 1) / (self.deform_levels + 1)))
            x1, y1, x2, y2 = get_center_crop_coords(self.t_w, self.t_h, window_w, window_h)
            one_hot_crop = self.one_hot[x1:x2, y1:y2]
            self.deform_hist_norms.append(one_hot_crop.sum(dim=(0, 1)) / (window_h * window_w))
            deform_filter = nn.Conv2d(
                self.n_clusters,
                self.n_clusters,
                (2, 2),
                dilation=(window_w, window_h),
                bias=False,
                groups=self.n_clusters,
            ).to(self.device)
            deform_filter.weight.data = (
                torch.tensor([[1, -1], [-1, 1]], device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            ).repeat(self.n_clusters, 1, 1, 1)
            self.deform_filters.append(deform_filter)
            self.deform_shapes.append((window_w, window_h))
            self.deform_weights[i + 1, 0] = 1 / (self.deform_levels + 1 - i)

        if filter_distances:
            max_dis_vals = [[] for _ in range(n_clusters)]
            for i in range(n_clusters):
                cluster_distances = dis_vals[torch.where(choice_cluster == i)[0], i]
                max_dis_vals[i] = cluster_distances.max().item()
            self.max_dis_vals = torch.tensor(max_dis_vals).to(self.device)
        else:
            self.max_dis_vals = torch.ones(n_clusters).to(self.device) * 1e10

    def get_heatmap(self, x):
        with torch.no_grad():
            one_hots, _ = self.get_nnf(x)

            integral_onehot = one_hots.cumsum(dim=1).cumsum(dim=2)

            query_area_histogram = self.sum_filter(integral_onehot.unsqueeze(0)).squeeze(0).permute(
                1, 2, 0
            ) / (self.t_w * self.t_h)
            orig_shape = query_area_histogram.shape[:2]

            overall_hist_diff = query_area_histogram - self.temp_area_hist_norm
            if self.hist_abs:
                overall_hist_diff = torch.abs(overall_hist_diff)

            deform_diff = self.get_deform_diff(integral_onehot, all_hist_diffs, orig_shape)

            all_hist_diffs = torch.cat(
                [overall_hist_diff.mean(-1).unsqueeze(-1), deform_diff], dim=-1
            )

            histogram_distance = (all_hist_diffs @ self.deform_weights).squeeze(-1)

        return histogram_distance

    def get_deform_diff(self, integral_onehot, all_hist_diffs, orig_shape):
        all_hist_diffs = []
        for i, deform_filter in enumerate(self.deform_filters):
            deform_histogram = deform_filter(integral_onehot.unsqueeze(0)).squeeze(0).permute(
                1, 2, 0
            ) / (self.deform_shapes[i][0] * self.deform_shapes[i][1])

            x1, y1, x2, y2 = get_center_crop_coords(
                deform_histogram.shape[0], deform_histogram.shape[1], orig_shape[0], orig_shape[1],
            )
            deform_histogram = deform_histogram[x1:x2, y1:y2, :]
            deform_hist_diff = deform_histogram - self.deform_hist_norms[i]
            if self.hist_abs:
                deform_hist_diff = torch.abs(deform_hist_diff)
            all_hist_diffs.append(deform_hist_diff.mean(-1).unsqueeze(-1))
        all_hist_diffs = torch.cat(all_hist_diffs, dim=-1)
        return all_hist_diffs

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
        # Filter the distance map
        one_hots[sim_one_hots > self.max_dis_vals.unsqueeze(-1).unsqueeze(-1)] = 0
        # print("Removed pixels: ", ((w * h) - one_hots.sum()).item())
        return one_hots, nnf_idxs  # adjacency_distance


class TemplateMatcher:
    def __init__(
        self,
        template,
        n_clusters=16,
        filter_distances=False,
        use_adj_weights=False,
        template_image=None,
        patch_loc=None,
    ):
        self.device = template.device
        self.kernel = torch.ones(3, 3).to(self.device)
        self.eps = 1e-6
        self.n_clusters = n_clusters
        self.n_chunks = 1000
        self.temp_dim_scale = 8
        self.hist_abs = True
        self.norm_hist_diff = False

        self.pool1d = nn.MaxPool1d(kernel_size=n_clusters, return_indices=True)
        self.unpool1d = nn.MaxUnpool1d(kernel_size=n_clusters)

        c, self.t_w, self.t_h = template.shape
        template_flatten = template.reshape(c, self.t_w * self.t_h).transpose(1, 0)
        # self.sum_filter = nn.Conv2d(
        #     1, 1, (self.t_w, self.t_h), bias=False  # , padding="same", padding_mode="reflect"
        # ).to(self.device)
        # self.sum_filter.weight.data = torch.ones(1, 1, self.t_w, self.t_h).to(self.device)

        self.sum_filter = nn.Conv2d(
            self.n_clusters,
            self.n_clusters,
            (2, 2),
            dilation=(self.t_w, self.t_h),
            bias=False,
            stride=1,
            groups=self.n_clusters,
        ).to(self.device)
        self.sum_filter.weight.data = (
            torch.tensor([[1, -1], [-1, 1]], device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        ).repeat(self.n_clusters, 1, 1, 1)

        self.cluster_centers, choice_cluster, dis_vals = kmeans(
            X=template_flatten, num_clusters=n_clusters
        )
        _, template_histogram = torch.unique(choice_cluster, return_counts=True)
        self.temp_area_hist_norm = template_histogram / template_histogram.sum()

        if filter_distances:
            max_dis_vals = [[] for _ in range(n_clusters)]
            for i in range(n_clusters):
                cluster_distances = dis_vals[torch.where(choice_cluster == i)[0], i]
                max_dis_vals[i] = cluster_distances.max().item()
            self.max_dis_vals = torch.tensor(max_dis_vals).to(self.device)
        else:
            self.max_dis_vals = torch.ones(n_clusters).to(self.device) * 1e10

        if use_adj_weights:
            temp_labels = choice_cluster.reshape(self.t_w, self.t_h)
            temp_labels_one_hot = (
                torch.nn.functional.one_hot(temp_labels, num_classes=n_clusters)
                .permute(2, 0, 1)
                .unsqueeze(1)
            )
            temp_labels_one_hot_dil = dilation(temp_labels_one_hot, self.kernel)

            temp_adj_mat = (
                temp_labels_one_hot_dil * temp_labels_one_hot_dil.permute(1, 0, 2, 3)
            ).sum(dim=(2, 3)) * (1 - torch.eye(n_clusters).to(self.device))
            temp_adj_mat_norm = temp_adj_mat / temp_adj_mat.sum(dim=1)
        else:
            temp_adj_mat_norm = None

        if template_image is not None:
            nnf_one_hot, nnf_idxs = self.get_nnf(template_image)
            query_area_hist_norm = (
                (self.sum_filter(nnf_one_hot.unsqueeze(1)) / (self.t_w * self.t_h))
                .squeeze(1)
                .permute(1, 2, 0)
            )
            histogram_diff = query_area_hist_norm - self.temp_area_hist_norm
            if self.hist_abs:
                histogram_diff = torch.abs(histogram_diff)
            # if self.norm_hist_diff:
            #     histogram_diff = histogram_diff / self.temp_area_hist_norm

            center_x, center_y = patch_loc[1] + patch_loc[3] // 2, patch_loc[0] + patch_loc[2] // 2
            width = self.t_w // self.temp_dim_scale
            height = self.t_h // self.temp_dim_scale

            temp_labels = torch.zeros_like(nnf_idxs)
            temp_labels[
                center_x - width // 2 : center_x + width // 2,
                center_y - height // 2 : center_y + height // 2,
            ] = 1
            bg_labels = torch.ones_like(nnf_idxs)
            bg_labels[
                center_x - self.t_w // 2 : center_x + self.t_w // 2,
                center_y - self.t_h // 2 : center_y + self.t_h // 2,
            ] = 0
            self.weight_calculator = WeightCalculator(n_clusters, self.device, temp_adj_mat_norm)
            self.weight_calculator.train_weights(histogram_diff, temp_labels, bg_labels)
            self.trained_weights = self.weight_calculator.get_learned_weights()
        else:
            self.trained_weights = None

    def get_heatmap(self, x):
        with torch.no_grad():
            one_hots, _ = self.get_nnf(x)

            integral_onehot = one_hots.cumsum(dim=1).cumsum(dim=2)

            query_area_histogram = self.sum_filter(integral_onehot.unsqueeze(0)).squeeze(0).permute(
                1, 2, 0
            ) / (self.t_w * self.t_h)

            # query_area_histogram = (
            #     (self.sum_filter(one_hots.unsqueeze(1)) / (self.t_w * self.t_h))
            #     .squeeze(1)
            #     .permute(1, 2, 0)
            # )

            histogram_diff = query_area_histogram - self.temp_area_hist_norm
            if self.hist_abs:
                histogram_diff = torch.abs(histogram_diff)
            if self.norm_hist_diff:
                histogram_diff = histogram_diff / self.temp_area_hist_norm

            if self.trained_weights is not None:
                histogram_distance = (histogram_diff @ self.trained_weights).squeeze(-1)
            else:
                histogram_distance = (histogram_diff).mean(dim=-1)

        return histogram_distance

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
        # Filter the distance map
        one_hots[sim_one_hots > self.max_dis_vals.unsqueeze(-1).unsqueeze(-1)] = 0
        # print("Removed pixels: ", ((w * h) - one_hots.sum()).item())
        return one_hots, nnf_idxs  # adjacency_distance


class TemplateMatcher2:
    def __init__(
        self, template, template_image, image_features, patch_loc, n_clusters=16, device="0"
    ):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

        self.kernel = torch.ones(3, 3).to(self.device)

        self.eps = 1e-6
        c, w, h = template.shape
        self.template_w = w
        self.template_h = h
        template_flatten = template.reshape(c, w * h).transpose(1, 0)

        self.cluster_centers, choice_cluster, dis_vals = kmeans(
            X=template_flatten, num_clusters=n_clusters
        )
        # min_dis_vals = dis_vals[torch.arange(0, len(choice_cluster)), choice_cluster]
        max_dis_vals = [[] for _ in range(n_clusters)]
        for i in range(n_clusters):
            cluster_distances = dis_vals[torch.where(choice_cluster == i)[0], i]
            max_dis_vals[i] = cluster_distances.max().item()
        self.max_dis_vals = torch.tensor(max_dis_vals).to(self.device)

        self.n_chunks = 1000
        _, template_histogram = torch.unique(choice_cluster, return_counts=True)
        self.template_area_histogram = template_histogram / template_histogram.sum()

        template_labels = choice_cluster.reshape(w, h)

        template_labels_one_hot = (
            torch.nn.functional.one_hot(template_labels, num_classes=n_clusters)
            .permute(2, 0, 1)
            .unsqueeze(1)
        )
        template_labels_one_hot_dil = dilation(template_labels_one_hot, self.kernel)

        fig, ax = plt.subplots(nrows=4, ncols=n_clusters // 4)
        i = 0
        for row in ax:
            for col in row:
                col.imshow(template_labels_one_hot[i, 0, :, :].cpu().numpy())
                i += 1

        template_adjacency_matrix = (
            template_labels_one_hot_dil * template_labels_one_hot_dil.permute(1, 0, 2, 3)
        ).sum(dim=(2, 3)) * (1 - torch.eye(n_clusters).to(self.device))
        template_adjacency_matrix = template_adjacency_matrix / template_adjacency_matrix.sum(dim=1)

        # plt.figure()
        # plt.imshow(template_adjacency_matrix.cpu().numpy())

        # template_adjacency_matrix1 = template_adjacency_matrix.clone()
        # template_adjacency_matrix1[template_adjacency_matrix1 > 0] = 1
        # template_adjacency_matrix1 = template_adjacency_matrix1.sum(dim=-1)
        # self.adjacency_weights = template_adjacency_matrix1 / template_adjacency_matrix1.sum()
        # # template_labels_ajcacency_sorted = template_adjacency_matrix.sort(descending=True)[0]
        # new_new_new_template_labels = torch.zeros_like(new_template_labels)
        # for i, label in enumerate(template_adjacency_matrix1):
        #     new_new_new_template_labels[template_labels == i] = label.long()

        self.pool1d = nn.MaxPool1d(kernel_size=n_clusters, return_indices=True)
        self.unpool1d = nn.MaxUnpool1d(kernel_size=n_clusters)
        self.sum_filter = nn.Conv2d(
            1,
            1,
            (w, h),
            padding="same",  # (w // 2 + 1, h // 2 + 1),
            padding_mode="reflect",
            # groups=n_clusters,
            bias=False,
        ).to(self.device)
        self.sum_filter.weight.data = torch.ones(1, 1, w, h).to(self.device)  # / (w * h)

        nnf_one_hot, nnf_idxs = self.get_nnf(template_image)
        query_area_histogram = (
            (self.sum_filter(nnf_one_hot.unsqueeze(1)) / (self.template_w * self.template_h))
            .squeeze(1)
            .permute(1, 2, 0)
        )
        histogram_diff = torch.abs(query_area_hist_norm - self.temp_area_hist_norm)

        labels = torch.zeros_like(nnf_idxs)
        center_x, center_y = patch_loc[1] + patch_loc[3] // 2, patch_loc[0] + patch_loc[2] // 2
        labels[
            center_x - self.template_w // 16 : center_x + self.template_w // 16,
            center_y - self.template_h // 16 : center_y + self.template_h // 16,
        ] = 1
        sample_weights = torch.zeros_like(nnf_idxs).float()
        u, c = torch.unique(nnf_idxs, return_counts=True)
        for i, j in zip(u, c):
            sample_weights[nnf_idxs == i] = c.max().item() / j

        self.weight_calculator = WeightCalculator(template_adjacency_matrix)
        self.popularity_weights = self.weight_calculator.train_weights(
            histogram_diff, sample_weights, labels
        )
        print(self.popularity_weights)
        # self.popularity_weights = c.sum() / c

        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(template_image)
        ax[1].imshow(template_labels.cpu().numpy(), cmap=cc.cm.glasbey)
        ax[2].imshow(template_image)
        ax[2].imshow(template_labels.cpu().numpy(), cmap=cc.cm.glasbey, alpha=0.5)
        # ax[3].imshow(image_nnf.cpu().numpy(), cmap=cc.cm.glasbey)

    def get_heatmap(self, x):
        with torch.no_grad():
            one_hots, _ = self.get_nnf(x)

            # weights = one_hots.sum() / (one_hots.sum(dim=(1, 2)) + 1)

            query_area_histogram = (
                (self.sum_filter(one_hots.unsqueeze(1)) / (self.template_w * self.template_h))
                .squeeze(1)
                .permute(1, 2, 0)
            )
            # query_area_histogram = query_area_histogram / query_area_histogram.sum(
            #     dim=-1, keepdim=True
            # )

            histogram_diff = (
                torch.abs(query_area_histogram - self.template_area_histogram)
                / self.template_area_histogram
            )
            histogram_distance = (histogram_diff).mean(dim=-1)

            histogram_distance1 = (histogram_diff @ self.popularity_weights).squeeze(-1)
        return histogram_distance, histogram_distance1

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
        # # Filter the distance map
        # one_hots[sim_one_hots > self.max_dis_vals.unsqueeze(-1).unsqueeze(-1)] = 0
        # print("Removed pixels: ", ((w * h) - one_hots.sum()).item())
        return one_hots, nnf_idxs  # adjacency_distance

        # template_labels = choice_cluster.reshape(w, h)
        # # # template_labels_area_sorted = template_histogram.sort(descending=True)[0]
        # new_template_labels = torch.zeros_like(template_labels)
        # for i, label in enumerate(template_histogram):
        #     new_template_labels[template_labels == i] = label

        # fig, ax = plt.subplots(nrows=1, ncols=3)
        # ax[0].imshow(template_image)
        # ax[1].imshow(new_template_labels.cpu().numpy(), cmap="viridis")
        # ax[2].imshow(template_image)
        # ax[2].imshow(new_template_labels.cpu().numpy(), cmap="viridis", alpha=0.4)

        # template_labels_one_hot = (
        #     torch.nn.functional.one_hot(template_labels, num_classes=n_clusters)
        #     .permute(2, 0, 1)
        #     .unsqueeze(1)
        # )
        # template_labels_perimeter = (
        #     (template_labels_one_hot - erosion(template_labels_one_hot, self.kernel))
        # ).sum(dim=(1, 2, 3))

        # self.template_area_perimeter_ratio = template_labels_perimeter / torch.sqrt(
        #     template_histogram
        # )

        # template_labels_one_hot_dil = dilation(template_labels_one_hot, self.kernel)
        # ##
        # template_one_hot_numpy = template_labels_one_hot_dil.squeeze(1).cpu().numpy()
        # n_connceted_regions = []
        # for i in range(n_clusters):
        #     n_connceted_regions.append(ndimage.label(template_one_hot_numpy[i])[1])

        # # template_labels_conn_sorted = torch.tensor(n_connceted_regions).sort(descending=True)[0]
        # new_new_template_labels = torch.zeros_like(template_labels)
        # for i, label in enumerate(torch.tensor(n_connceted_regions)):
        #     new_new_template_labels[template_labels == i] = label

        # fig, ax = plt.subplots(nrows=1, ncols=3)
        # ax[0].imshow(template_image)
        # ax[1].imshow(new_new_template_labels.cpu().numpy(), cmap="viridis")
        # ax[2].imshow(
        #     np.log(1 / (new_template_labels * new_new_template_labels).cpu().numpy()),
        #     cmap="viridis",
        # )
        # # ax[2].imshow(template_image)
        # # ax[2].imshow(new_new_template_labels.cpu().numpy(), cmap="viridis", alpha=0.4)

        # template_labels_one_hot_open = dilation(
        #     opening(template_labels_one_hot, self.kernel), self.kernel
        # )

        # template_adjacency_matrix = (
        #     template_labels_one_hot_open * template_labels_one_hot_open.permute(1, 0, 2, 3)
        # ).sum(dim=(2, 3)) * (1 - torch.eye(n_clusters).to(self.device))

        # template_adjacency_matrix1 = template_adjacency_matrix.clone()
        # template_adjacency_matrix1[template_adjacency_matrix1 > 0] = 1
        # template_adjacency_matrix1 = template_adjacency_matrix1.sum(dim=-1)
        # self.adjacency_weights = template_adjacency_matrix1 / template_adjacency_matrix1.sum()
        # # template_labels_ajcacency_sorted = template_adjacency_matrix.sort(descending=True)[0]
        # new_new_new_template_labels = torch.zeros_like(new_template_labels)
        # for i, label in enumerate(template_adjacency_matrix1):
        #     new_new_new_template_labels[template_labels == i] = label.long()

        # ax[1].imshow(new_new_new_template_labels.cpu().numpy(), cmap="viridis")
        # ax[2].imshow(
        #     # np.log(
        #     ((1 / new_template_labels) * (new_new_template_labels)).cpu().numpy(),  # )
        #     cmap="viridis",
        # )
        # # ax[2].imshow(template_image)
        # # ax[2].imshow(new_new_new_template_labels.cpu().numpy(), cmap="viridis", alpha=0.4)

        # self.template_adjacency_matrix = template_adjacency_matrix / template_adjacency_matrix.sum(
        #     dim=1, keepdim=True
        # )

        # self.pool1d = nn.MaxPool1d(kernel_size=n_clusters, return_indices=True)
        # self.unpool1d = nn.MaxUnpool1d(kernel_size=n_clusters)
        # self.sum_filter = nn.Conv2d(
        #     1,
        #     1,
        #     (w, h),
        #     padding="same",  # (w // 2 + 1, h // 2 + 1),
        #     padding_mode="reflect",
        #     # groups=n_clusters,
        #     bias=False,
        # ).to(self.device)
        # self.sum_filter.weight.data = torch.ones(1, 1, w, h).to(self.device)  # / (w * h)

        # # self.sum_filter = nn.Conv2d(
        # #     n_clusters,
        # #     n_clusters,
        # #     (w, h),
        # #     padding="same",  # (w // 2 + 1, h // 2 + 1),
        # #     padding_mode="reflect",
        # #     groups=n_clusters,
        # #     bias=False,
        # # ).to(self.device)
        # # self.sum_filter.weight.data = dilation(template_labels_one_hot, self.kernel)  # / (w * h)

        # # print(dis_vals.min(), dis_vals.max())
        # # print(choice_cluster.shape, cluster_centers.shape)
        # # feature_histogram = counts  # / (w * h)

        # # dist_calculator = nn.Conv1d(
        # #     in_channels=c, out_channels=n_clusters, kernel_size=1, bias=False
        # # ).to(feature_extractor.device)
        # # dist_calculator.weight.data = cluster_centers.unsqueeze(-1).to(torch.float32)
        # # dist_calculator.weight.requires_grad = False

        # self.template = template

        # one_hot_dilated = dilation(one_hots.unsqueeze(1), self.kernel)
        # one_hot_adjacency_matrix = one_hot_dilated * one_hot_dilated.permute(1, 0, 2, 3)
        # c, c, w, h = one_hot_adjacency_matrix.shape
        # query_adjacency_matrix_sum = (
        #     self.sum_filter(one_hot_adjacency_matrix.reshape(c * c, 1, w, h))
        #     .reshape(c, c, w, h)
        #     .permute(2, 3, 0, 1)
        # ) * (1 - torch.eye(c).to(self.device))
        # query_adjacency_matrix_sum = query_adjacency_matrix_sum / (
        #     query_adjacency_matrix_sum.sum(dim=-1, keepdim=True) + self.eps
        # )

        # query_adjacency_matrix = torch.zeros_like(query_adjacency_matrix_sum)
        # query_adjacency_matrix[query_adjacency_matrix_sum > 0.1] = 1

        # adjacency_matrix_distance = (
        #     torch.abs(query_adjacency_matrix - self.template_adjacency_matrix)
        #     .sum(dim=-1)
        #     .mean(dim=-1)
        # )

        # query_perimeter_histogram = (
        #     self.sum_filter(one_hots - erosion(one_hots, self.kernel))
        #     .squeeze(1)
        #     .permute(1, 2, 0)
        # )
        # query_area_perimeter_ratio = query_perimeter_histogram / torch.sqrt(
        #     query_area_histogram + self.eps
        # )
        # histogram_distance = (
        #     torch.abs(query_area_perimeter_ratio - self.template_area_perimeter_ratio)
        #     / (self.template_area_perimeter_ratio + self.eps)
        # ).mean(dim=-1)

        # w, h, c = query_area_histogram.shape
        # # calculation = ddks.methods.ddKS()
        # # distance = calculation(
        # #     query_area_histogram.reshape(w * h, c), self.template_area_histogram.unsqueeze(0)
        # # )
        # print(f"The ddKS distance is {distance}")

        # one_hot_dilated = dilation(one_hots, self.kernel)
        # one_hot_adjacency_matrix = one_hot_dilated * one_hot_dilated.permute(1, 0, 2, 3)
        # c, c, w, h = one_hot_adjacency_matrix.shape
        # one_hot_adjacency_matrix = (
        #     self.sum_filter(one_hot_adjacency_matrix.reshape(c * c, 1, w, h))
        #     .reshape(c, c, w, h)
        #     .permute(2, 3, 0, 1)
        # ) * (1 - torch.eye(c).to(self.device))
        # query_adjacency_matrix = one_hot_adjacency_matrix / (
        #     one_hot_adjacency_matrix.sum(dim=-1, keepdim=True) + self.eps
        # )

        # adjacency_distance = torch.abs(
        #     query_adjacency_matrix - self.template_adjacency_matrix
        # ).sum(dim=(2, 3))
        # kl_dis = self.kl_div(
        #     torch.log(query_adjacency_matrix + self.eps), self.template_adjacency_matrix
        # ).sum(dim=(2, 3))

        # one_hots = dilation(one_hots, self.kernel)
        # # one_hots1 = one_hots.to_sparse()
        # # one_hots_t = one_hots.permute(1, 0, 2, 3).to_sparse()

        # one_hot_adjacency_matrix = one_hots * one_hots.permute(1, 0, 2, 3)
        # c, c, w, h = one_hot_adjacency_matrix.shape
        # sum_hist = (self.sum_filter(one_hot_adjacency_matrix.reshape(c * c, 1, w, h))).reshape(
        #     c, c, w, h
        # )

        # tmp_adj_mat = self.template_adjacency_matrix.unsqueeze(-1).unsqueeze(-1)
        # histogram_mask = torch.eye(c).to(self.device).unsqueeze(-1).unsqueeze(-1)
        # score = (torch.abs(sum_hist - tmp_adj_mat)).sum(dim=(0, 1))

        # max_dist_mat = max_dis_per_cluster[:, None, None].repeat(1, w, h)
        # print(one_hots.sum())
        # one_hots[sim_one_hots > (5 * max_dist_mat)] = 0
        # print(one_hots.sum())


class TemplateMatcher2:
    def __init__(self, template, template_image, n_clusters=16, device="0"):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")

        self.kernel = torch.ones(3, 3).to(self.device)

        self.eps = 1e-6
        c, w, h = template.shape
        self.template_w = w
        self.template_h = h
        template_flatten = template.reshape(c, w * h).transpose(1, 0)

        self.cluster_centers, choice_cluster, dis_vals = kmeans(
            X=template_flatten, num_clusters=n_clusters
        )
        min_dis_vals = dis_vals[torch.arange(0, len(choice_cluster)), choice_cluster]
        dis_vals_cluster = [[] for _ in range(n_clusters)]
        for i, cluster in enumerate(choice_cluster):
            dis_vals_cluster[cluster].append(min_dis_vals[i].item())
        ssd = []
        max_dis_vals = []
        for i in range(n_clusters):
            ssd.append(np.mean(dis_vals_cluster[i]))
            max_dis_vals.append(max(dis_vals_cluster[i]))
        self.max_dis_vals = torch.tensor(max_dis_vals).to(self.device)
        self.ssd_weight = (
            torch.tensor(ssd).to(self.device) / torch.tensor(ssd).to(self.device).max()
        )

        self.n_chunks = 1000
        _, template_histogram = torch.unique(choice_cluster, return_counts=True)
        self.template_area_histogram = template_histogram / template_histogram.sum()

        # self.template_log_histogram = torch.log(self.template_area_histogram)

        template_labels = choice_cluster.reshape(w, h)
        # # template_labels_area_sorted = template_histogram.sort(descending=True)[0]
        new_template_labels = torch.zeros_like(template_labels)
        for i, label in enumerate(template_histogram):
            new_template_labels[template_labels == i] = label

        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(template_image)
        ax[1].imshow(new_template_labels.cpu().numpy(), cmap="viridis")
        ax[2].imshow(template_image)
        ax[2].imshow(new_template_labels.cpu().numpy(), cmap="viridis", alpha=0.4)

        template_labels_one_hot = (
            torch.nn.functional.one_hot(template_labels, num_classes=n_clusters)
            .permute(2, 0, 1)
            .unsqueeze(1)
        )
        template_labels_perimeter = (
            (template_labels_one_hot - erosion(template_labels_one_hot, self.kernel))
        ).sum(dim=(1, 2, 3))

        self.template_area_perimeter_ratio = template_labels_perimeter / torch.sqrt(
            template_histogram
        )

        template_labels_one_hot_dil = dilation(template_labels_one_hot, self.kernel)
        ##
        template_one_hot_numpy = template_labels_one_hot_dil.squeeze(1).cpu().numpy()
        n_connceted_regions = []
        for i in range(n_clusters):
            n_connceted_regions.append(ndimage.label(template_one_hot_numpy[i])[1])

        # template_labels_conn_sorted = torch.tensor(n_connceted_regions).sort(descending=True)[0]
        new_new_template_labels = torch.zeros_like(template_labels)
        for i, label in enumerate(torch.tensor(n_connceted_regions)):
            new_new_template_labels[template_labels == i] = label

        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(template_image)
        ax[1].imshow(new_new_template_labels.cpu().numpy(), cmap="viridis")
        ax[2].imshow(
            np.log(1 / (new_template_labels * new_new_template_labels).cpu().numpy()),
            cmap="viridis",
        )
        # ax[2].imshow(template_image)
        # ax[2].imshow(new_new_template_labels.cpu().numpy(), cmap="viridis", alpha=0.4)

        template_labels_one_hot_open = dilation(
            opening(template_labels_one_hot, self.kernel), self.kernel
        )

        template_adjacency_matrix = (
            template_labels_one_hot_open * template_labels_one_hot_open.permute(1, 0, 2, 3)
        ).sum(dim=(2, 3)) * (1 - torch.eye(n_clusters).to(self.device))

        template_adjacency_matrix1 = template_adjacency_matrix.clone()
        template_adjacency_matrix1[template_adjacency_matrix1 > 0] = 1
        template_adjacency_matrix1 = template_adjacency_matrix1.sum(dim=-1)
        self.adjacency_weights = template_adjacency_matrix1 / template_adjacency_matrix1.sum()
        # template_labels_ajcacency_sorted = template_adjacency_matrix.sort(descending=True)[0]
        new_new_new_template_labels = torch.zeros_like(new_template_labels)
        for i, label in enumerate(template_adjacency_matrix1):
            new_new_new_template_labels[template_labels == i] = label.long()

        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(template_image)
        ax[1].imshow(new_new_new_template_labels.cpu().numpy(), cmap="viridis")
        ax[2].imshow(
            # np.log(
            ((1 / new_template_labels) * (new_new_template_labels)).cpu().numpy(),  # )
            cmap="viridis",
        )
        # ax[2].imshow(template_image)
        # ax[2].imshow(new_new_new_template_labels.cpu().numpy(), cmap="viridis", alpha=0.4)

        self.template_adjacency_matrix = template_adjacency_matrix / template_adjacency_matrix.sum(
            dim=1, keepdim=True
        )

        self.pool1d = nn.MaxPool1d(kernel_size=n_clusters, return_indices=True)
        self.unpool1d = nn.MaxUnpool1d(kernel_size=n_clusters)
        self.sum_filter = nn.Conv2d(
            1,
            1,
            (w, h),
            padding="same",  # (w // 2 + 1, h // 2 + 1),
            padding_mode="reflect",
            # groups=n_clusters,
            bias=False,
        ).to(self.device)
        self.sum_filter.weight.data = torch.ones(1, 1, w, h).to(self.device)  # / (w * h)

        # self.sum_filter = nn.Conv2d(
        #     n_clusters,
        #     n_clusters,
        #     (w, h),
        #     padding="same",  # (w // 2 + 1, h // 2 + 1),
        #     padding_mode="reflect",
        #     groups=n_clusters,
        #     bias=False,
        # ).to(self.device)
        # self.sum_filter.weight.data = dilation(template_labels_one_hot, self.kernel)  # / (w * h)

        fig, ax = plt.subplots(nrows=4, ncols=4)
        i = 0
        for row in ax:
            for col in row:
                col.imshow(template_labels_one_hot[i, 0, :, :].cpu().numpy())
                i += 1
        # print(dis_vals.min(), dis_vals.max())
        # print(choice_cluster.shape, cluster_centers.shape)
        # feature_histogram = counts  # / (w * h)

        # dist_calculator = nn.Conv1d(
        #     in_channels=c, out_channels=n_clusters, kernel_size=1, bias=False
        # ).to(feature_extractor.device)
        # dist_calculator.weight.data = cluster_centers.unsqueeze(-1).to(torch.float32)
        # dist_calculator.weight.requires_grad = False

        self.template = template

    def get_heatmap(self, x):
        with torch.no_grad():
            d, w, h = x.shape
            # print(d, w, h)
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

            nnf_idxs = torch.cat(nnf_idxs).transpose(0, 1).reshape(1, w, h)
            sim_vals = -torch.cat(sim_vals).transpose(0, 1).reshape(1, w, h)
            sim_one_hots = -torch.cat(sim_one_hots).transpose(0, 1).reshape(-1, w, h)
            one_hots = torch.cat(one_hots).transpose(0, 1).reshape(-1, w, h)

            one_hots[sim_one_hots > self.max_dis_vals.unsqueeze(-1).unsqueeze(-1)] = 0
            print("Removed pixels: ", ((w * h) - one_hots.sum()).item())
            one_hots = one_hots.unsqueeze(1)
            query_area_histogram = (
                (self.sum_filter(one_hots) / (self.template_w * self.template_h))
                .squeeze(1)
                .permute(1, 2, 0)
            )
            # query_perimeter_histogram = (
            #     self.sum_filter(one_hots - erosion(one_hots, self.kernel))
            #     .squeeze(1)
            #     .permute(1, 2, 0)
            # )
            # query_area_perimeter_ratio = query_perimeter_histogram / torch.sqrt(
            #     query_area_histogram + self.eps
            # )
            # histogram_distance = (
            #     torch.abs(query_area_perimeter_ratio - self.template_area_perimeter_ratio)
            #     / (self.template_area_perimeter_ratio + self.eps)
            # ).mean(dim=-1)

            # w, h, c = query_area_histogram.shape
            # # calculation = ddks.methods.ddKS()
            # # distance = calculation(
            # #     query_area_histogram.reshape(w * h, c), self.template_area_histogram.unsqueeze(0)
            # # )
            # print(f"The ddKS distance is {distance}")

            histogram_diff = torch.abs(query_area_histogram - self.template_area_histogram)
            histogram_distance = (histogram_diff / self.template_area_histogram).mean(dim=-1)

            histogram_distance1 = (histogram_diff * self.ssd_weight).sum(dim=-1)

            # one_hot_dilated = dilation(one_hots, self.kernel)
            # one_hot_adjacency_matrix = one_hot_dilated * one_hot_dilated.permute(1, 0, 2, 3)
            # c, c, w, h = one_hot_adjacency_matrix.shape
            # one_hot_adjacency_matrix = (
            #     self.sum_filter(one_hot_adjacency_matrix.reshape(c * c, 1, w, h))
            #     .reshape(c, c, w, h)
            #     .permute(2, 3, 0, 1)
            # ) * (1 - torch.eye(c).to(self.device))
            # query_adjacency_matrix = one_hot_adjacency_matrix / (
            #     one_hot_adjacency_matrix.sum(dim=-1, keepdim=True) + self.eps
            # )

            # adjacency_distance = torch.abs(
            #     query_adjacency_matrix - self.template_adjacency_matrix
            # ).sum(dim=(2, 3))
            # kl_dis = self.kl_div(
            #     torch.log(query_adjacency_matrix + self.eps), self.template_adjacency_matrix
            # ).sum(dim=(2, 3))

            # one_hots = dilation(one_hots, self.kernel)
            # # one_hots1 = one_hots.to_sparse()
            # # one_hots_t = one_hots.permute(1, 0, 2, 3).to_sparse()

            # one_hot_adjacency_matrix = one_hots * one_hots.permute(1, 0, 2, 3)
            # c, c, w, h = one_hot_adjacency_matrix.shape
            # sum_hist = (self.sum_filter(one_hot_adjacency_matrix.reshape(c * c, 1, w, h))).reshape(
            #     c, c, w, h
            # )

            # tmp_adj_mat = self.template_adjacency_matrix.unsqueeze(-1).unsqueeze(-1)
            # histogram_mask = torch.eye(c).to(self.device).unsqueeze(-1).unsqueeze(-1)
            # score = (torch.abs(sum_hist - tmp_adj_mat)).sum(dim=(0, 1))

            # max_dist_mat = max_dis_per_cluster[:, None, None].repeat(1, w, h)
            # print(one_hots.sum())
            # one_hots[sim_one_hots > (5 * max_dist_mat)] = 0
            # print(one_hots.sum())
        return histogram_distance, histogram_distance1  # adjacency_distance


def max_locs(
    x,
    cluster_centers,
    sum_dis_filter,
    feature_norm,
    pool1d,
    unpool1d,
    feature_histogram,
    max_dis_per_cluster,
    n_chunks,
):
    with torch.no_grad():
        # print(x_norm.shape)
        sum_dis_filter.eval()
        d, w, h = x.shape
        # print(d, w, h)
        chunks = torch.split(x.reshape(d, w * h).transpose(1, 0), n_chunks, dim=0)

        nnf_idxs = []
        sim_vals = []
        one_hots = []
        sim_one_hots = []
        for i, chunk in enumerate(chunks):
            dist = -torch.cdist(cluster_centers, chunk)
            sim, idx = pool1d(dist.transpose(0, 1))
            one_hot = torch.ones_like(sim)
            one_hots.append(unpool1d(one_hot, idx))
            sim_one_hots.append(unpool1d(sim, idx))
            sim_vals.append(sim)
            nnf_idxs.append(idx)

        nnf_idxs = torch.cat(nnf_idxs).transpose(0, 1).reshape(1, w, h)
        sim_vals = -torch.cat(sim_vals).transpose(0, 1).reshape(1, w, h)
        sim_one_hots = -torch.cat(sim_one_hots).transpose(0, 1).reshape(-1, w, h)

        one_hots = torch.cat(one_hots).transpose(0, 1).reshape(-1, w, h)
        max_dist_mat = max_dis_per_cluster[:, None, None].repeat(1, w, h)
        print(one_hots.sum())
        one_hots[sim_one_hots > (5 * max_dist_mat)] = 0
        print(one_hots.sum())
        sum_hist = sum_dis_filter(one_hots.unsqueeze(0))
        a = (
            feature_histogram.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, sum_hist.shape[2], sum_hist.shape[3])
        )
        # print(a.shape, sum_hist.shape)
        kl = torch.abs(a - sum_hist).mean(dim=1)  # / (w * h)
        # kl = -torch.sum(torch.min(a, sum_hist), dim=1)
        # kl = torch.kl_div(sum_hist, a, log_target=True).mean(dim=1)
        # kl = -torch.log(torch.sum(torch.sqrt(torch.abs(torch.mul(a, sum_hist))), dim = 1)+1e-40)#
        idxs = torch.where(kl == kl.max())
        # print(a.min(), a.max(), sum_hist.min(), sum_hist.max(), kl.min(), kl.max())

        # sim_one_hots = torch.cat(sim_one_hots).transpose(0, 1).reshape(-1, w, h)
        # sim_weight_hist = sum_dis_filter(sim_one_hots.unsqueeze(0)) / num_vectors

        # unique_hist1 = dis_calculation(one_hots)
    return -kl, idxs
