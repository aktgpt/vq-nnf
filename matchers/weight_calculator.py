import torch
from torch import nn
from kmeans_pytorch import kmeans, bisecting_kmeans
import matplotlib.pyplot as plt
from kornia.morphology import dilation, erosion, opening
import torch.nn.functional as F
from scipy import ndimage
import numpy as np
import ddks
import colorcet as cc


def tanh_scaled(x):
    x_tanh = torch.tanh(x)
    x_tanh_scaled = (1 + x_tanh) / 2
    return x_tanh_scaled


class WeightCalculator:
    def __init__(self, channels, device, temp_adj_matrix, margin=0.2):
        self.channels = channels
        self.device = device
        self.margin = margin
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        if not temp_adj_matrix is None:
            self.temp_adj_matrix = temp_adj_matrix
            self.adj_weights = torch.ones(channels, device=self.device)
            self.adj_weights_calc = nn.Parameter(
                torch.ones((channels, channels), requires_grad=True, device=self.device) / channels
            )
            self.adj_area_weight = nn.Parameter(
                torch.ones((2, 1), requires_grad=True, device=self.device)
            )
        else:
            self.temp_adj_matrix = None

        # self.area_weights = torch.ones(channels, device=self.device)
        self.area_weights_calc = nn.Sequential(
            nn.Conv2d(self.channels, self.channels // 8, 1, bias=False),
            nn.BatchNorm2d(self.channels // 8),
            nn.ReLU(),
            nn.Conv2d(self.channels // 8, self.channels, 1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(),
        )

    # def get_adjancency_weights(self):
    #     # with torch.no_grad:
    #     x = self.temp_adj_matrix * self.adj_weights_calc
    #     self.adj_weights = x.sum(dim=1).unsqueeze(1)

    def train_weights(self, train_dataset, temp_labels, bg_labels, n_iters=10):
        optimizer = torch.optim.AdamW(self.area_weights_calc.parameters(), lr=1e-3)
        self.area_weights_calc.train()
        for i in range(n_iters):
            optimizer.zero_grad()
            weights = self.area_weights_calc(train_dataset)
            loss = self.get_loss(train_dataset, weights, temp_labels, bg_labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            print(f"Iteration: {i}, Loss: {loss.item()}")

    def get_loss(self, train_histogram, weights, temp_labels, bg_labels):
        op_distance = (train_histogram @ weights).squeeze(-1)

        label_loss = (
            torch.max(torch.zeros_like(op_distance), op_distance) * temp_labels  # * sample_weight
        ).sum() / temp_labels.sum()
        bg_labels_loss = (
            torch.max(torch.zeros_like(op_distance), self.margin - op_distance) * bg_labels
        ).sum() / bg_labels.sum()

        loss = label_loss + bg_labels_loss
        return loss

    def get_learned_weights(self):
        return self.area_weights_calc


class WeightCalculator1:
    def __init__(self, channels, device, temp_adj_matrix, margin=0.2):
        self.channels = channels
        self.device = device
        self.margin = margin
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        if not temp_adj_matrix is None:
            self.temp_adj_matrix = temp_adj_matrix
            self.adj_weights = torch.ones(channels, device=self.device)
            self.adj_weights_calc = nn.Parameter(
                torch.ones((channels, channels), requires_grad=True, device=self.device) / channels
            )
            self.adj_area_weight = nn.Parameter(
                torch.ones((2, 1), requires_grad=True, device=self.device)
            )
        else:
            self.temp_adj_matrix = None

        self.area_weights = torch.ones(channels, device=self.device)
        self.area_weights_calc = nn.Parameter(
            torch.ones(channels, 1, requires_grad=True, device=self.device) / channels
        )

    def get_adjancency_weights(self):
        # with torch.no_grad:
        x = self.temp_adj_matrix * self.adj_weights_calc
        self.adj_weights = x.sum(dim=1).unsqueeze(1)

    def get_area_weights(self):
        # with torch.no_grad:
        self.area_weights = self.area_weights_calc

    def train_weights(self, train_dataset, temp_labels, bg_labels, n_iters=10):
        optimizer = torch.optim.AdamW(
            [self.adj_weights_calc, self.adj_area_weight, self.area_weights_calc]
            if self.temp_adj_matrix is not None
            else [self.area_weights_calc],
            lr=0.5,
        )
        for i in range(n_iters):
            optimizer.zero_grad()
            loss = self.get_loss(train_dataset, temp_labels, bg_labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            print(f"Iteration: {i}, Loss: {loss.item()}")

    def get_loss(self, train_histogram, temp_labels, bg_labels):
        weights = self.get_learned_weights()

        op_distance = (train_histogram @ weights).squeeze(-1)

        label_loss = (
            torch.max(torch.zeros_like(op_distance), op_distance) * temp_labels  # * sample_weight
        ).sum() / temp_labels.sum()
        bg_labels_loss = (
            torch.max(torch.zeros_like(op_distance), self.margin - op_distance) * bg_labels
        ).sum() / bg_labels.sum()

        loss = label_loss + bg_labels_loss
        return loss

    def get_learned_weights(self):
        self.get_area_weights()

        if self.temp_adj_matrix is not None:
            self.get_adjancency_weights()
            weights = torch.cat([self.adj_weights, self.area_weights], dim=1)
            weights = self.relu(weights @ self.adj_area_weight)
        else:
            weights = self.relu(self.area_weights)
        weights = weights / weights.sum()
        return weights


class WeightCalculator2:
    def __init__(self, template_adjacency_matrix):
        self.device = template_adjacency_matrix.device
        c, c = template_adjacency_matrix.shape

        self.edge_weights = torch.ones(c, device=self.device)
        self.histogram_weights = torch.ones(c, device=self.device)

        self.template_adjacency_matrix = template_adjacency_matrix
        self.edge_weight_calculator = nn.Parameter(
            torch.ones((c, c), requires_grad=True, device=self.device) / c
        )  # .to(self.device)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0).to(self.device)
        self.tanh = nn.Tanh().to(self.device)

        self.histogram_weight_calculator = nn.Parameter(
            torch.ones(c, 1, requires_grad=True, device=self.device) / c
        )  # .to(self.device)
        self.margin = 2.0

        self.weight_weight = nn.Parameter(
            torch.ones((2, 1), requires_grad=True, device=self.device)
        )

    def get_adjancency_weights(self):
        x = self.template_adjacency_matrix * self.edge_weight_calculator
        self.edge_weights = x.sum(dim=1).unsqueeze(1)
        # return self.edge_weights

    def get_histogram_weights(self):
        self.histogram_weights = self.histogram_weight_calculator
        # return self.histogram_weights

    def train_weights(self, train_histogram, sample_weight, labels, n_iters=10):
        optimizer = torch.optim.AdamW(
            [self.edge_weight_calculator, self.histogram_weight_calculator, self.weight_weight],
            lr=0.5,
        )
        for i in range(n_iters):
            optimizer.zero_grad()
            loss = self.get_loss(train_histogram, sample_weight, labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            print(
                f"Iteration: {i}, Loss: {loss.item()},  {self.histogram_weight_calculator.grad.max().item()}"
            )
        final_weights = self.relu(self.histogram_weights)
        # tanh_scaled(
        #     # (self.edge_weights * torch.sigmoid(self.weight_weight))
        #     # + (self.histogram_weights * (1 - torch.sigmoid(self.weight_weight)))
        #     torch.cat([self.edge_weights, self.histogram_weights], dim=1)
        #     @ self.tanh(self.weight_weight)
        # )
        final_weights = final_weights / final_weights.sum()
        # print(self.softmax(self.weight_weight))
        return final_weights

    def get_loss(self, train_histogram, sample_weight, labels):
        self.get_adjancency_weights()
        self.get_histogram_weights()

        weights = self.relu(self.histogram_weights)
        # weights = tanh_scaled(
        #     # (self.edge_weights * torch.sigmoid(self.weight_weight))
        #     # + (self.histogram_weights * (1 - torch.sigmoid(self.weight_weight)))
        #     torch.cat([self.edge_weights, self.histogram_weights], dim=1)
        #     @ self.softmax(self.weight_weight)
        # )
        # weights = tanh_scaled(self.edge_weights + self.histogram_weights)
        # print(
        #     f"Weights Similarity: {F.cosine_similarity(self.edge_weights.permute(1,0), self.histogram_weights.permute(1,0)).item()}\
        #     {F.cosine_similarity(weights.permute(1,0), self.histogram_weights.permute(1,0)).item()}\
        #     {F.cosine_similarity(weights.permute(1,0), self.edge_weights.permute(1,0)).item()}"
        # )
        weights = weights / weights.sum()

        op_distance = torch.abs(train_histogram @ weights).squeeze(-1)

        label_loss = (
            torch.max(torch.zeros_like(op_distance), op_distance) * labels  # * sample_weight
        ).sum() / labels.sum()

        not_labels = 1 - labels
        not_label_loss = (
            torch.max(torch.zeros_like(op_distance), self.margin - op_distance)
            * not_labels
            # * sample_weight
        ).sum() / not_labels.sum()

        loss = label_loss + not_label_loss

        return loss

