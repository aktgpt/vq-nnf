import argparse
import os
import random
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc
from itertools import combinations, product
from scipy import stats


sns.set_style("whitegrid")
sns.set_context("talk")  # paper, notebook, talk, poster
folder = "exps_final"
plt.rcParams["font.family"] = "Arial"


methods = ["DDIS", "DIWU"]
n_features = [27, 512]
comp_df = []
for method in methods:
    for n_feature in n_features:
        df = pd.read_csv(f"exps_final/comp/tlpattr/TLPattr_{n_feature}_{method}_results.csv")
        df["Method"] = method
        df["Features"] = n_feature
        comp_df.append(df)

comp_df = pd.concat(comp_df).reset_index(drop=True)
comp_df = comp_df.rename(
    columns={
        "rotation": "exp_scales",
        "mious": "MIoU",
        "sr": "Success Rate",
        "avg_time": "Time (s)",
    }
)


result_csvs = []
n_clusters = [128]
n_haar_features = [1, 2, 3, 23]
for n_feat in n_features:
    result_csv = pd.read_csv(
        f"exps_final/tlpattr/scale/tlpattr_dataset_all_time_results_k3s2haar_scale_{n_feat}.csv"
    )
    method = (
        f"Scales:"
        + result_csv["scales"].astype(str)
        + ", Haar:"
        + result_csv["n_haar_features"].astype(str)
    )
    methods.extend(np.unique(method).tolist())

    result_csv["Method"] = method
    result_csv["Time (s)"] = result_csv["Temp_Match_Time"] + result_csv["Kmeans_Time"]
    result_csvs.append(result_csv)


result_csvs = pd.concat(result_csvs).reset_index(drop=True)[
    ["exp_scales", "M_IOU", "Success_Rate", "Method", "Time (s)", "n_features"]
]
result_csvs = result_csvs.rename(
    columns={"M_IOU": "MIoU", "Success_Rate": "Success Rate", "n_features": "Features",}
)


all_df = pd.concat([comp_df, result_csvs]).reset_index(drop=True)
all_df = all_df.rename(columns={"exp_scales": "Scale Factor"})

for n_feat in n_features:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    g = sns.lineplot(
        x="Scale Factor",
        y="MIoU",
        hue="Method",
        data=all_df.loc[all_df["Features"] == n_feat],
        palette="colorblind",
        alpha=1.0,
        ax=ax,
        marker="o",
    )
    # highlight the best result with a horizontal line
    max_miou = all_df.loc[all_df["Features"] == n_feat, "MIoU"].max()
    g.axhline(max_miou, ls="--", color="k", alpha=0.5, label=f"Best MIoU: {max_miou:.3f}")
    g.legend(loc="lower right")

    # plt.xticks(np.arange(0.2, 1.1, 0.2))
    plt.title(f"Feature Dimensions: {n_feat}")
    plt.savefig(f"exps_final/figures/scale/scale_comp_{n_feat}.pdf", bbox_inches="tight", dpi=500)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    g = sns.lineplot(
        x="Scale Factor",
        y="Time (s)",
        hue="Method",
        # style="Method",
        data=all_df.loc[all_df["Features"] == n_feat],
        palette="colorblind",
        alpha=1.0,
        ax=ax,
        marker="o",
    )
    # plt.yscale("log")
    plt.title(f"Feature Dimensions: {n_feat}")
    plt.savefig(f"exps_final/figures/scale/scale_time_{n_feat}.pdf", bbox_inches="tight", dpi=500)
    plt.close("all")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    g = sns.lineplot(
        x="Scale Factor",
        y="Time (s)",
        hue="Method",
        # style="Method",
        data=all_df.loc[all_df["Features"] == n_feat],
        palette="colorblind",
        alpha=1.0,
        ax=ax,
        marker="o",
    )
    plt.ylim(0, 2)  # 10 if n_feat == 512 else 2)
    plt.title(f"Feature Dimensions: {n_feat}")
    plt.savefig(
        f"exps_final/figures/scale/scale_time_{n_feat}_zoom.pdf", bbox_inches="tight", dpi=500
    )
    plt.close("all")


# plt.show()
x = 1

# wilcoxon_results_mat = np.zeros(
#     (len(scale_haar_feat_combinations), len(scale_haar_feat_combinations))
# )
# for n_clus in n_clusters:
#     df = result_csv.loc[result_csv["n_clusters"] == n_clus]

#     for i, (scale, n_haar_feat) in enumerate(scale_haar_feat_combinations):
#         for j, (scale2, n_haar_feat2) in enumerate(scale_haar_feat_combinations):
#             if not (scale == scale2 and n_haar_feat == n_haar_feat2):
#                 print(
#                     f"Wilcoxon signed rank test for scale {scale} and {scale2} and n_haar_feat {n_haar_feat} and {n_haar_feat2}"
#                 )
#                 print(
#                     stats.mannwhitneyu(
#                         df.loc[
#                             (df["scales"] == scale) & (df["n_haar_features"] == n_haar_feat),
#                             "M_IOU",
#                         ].values,
#                         df.loc[
#                             (df["scales"] == scale2) & (df["n_haar_features"] == n_haar_feat2),
#                             "M_IOU",
#                         ].values,
#                         alternative="greater",
#                     )
#                 )
#                 wilcoxon_results_mat[i, j] = stats.mannwhitneyu(
#                     df.loc[
#                         (df["scales"] == scale) & (df["n_haar_features"] == n_haar_feat), "M_IOU",
#                     ].values,
#                     df.loc[
#                         (df["scales"] == scale2) & (df["n_haar_features"] == n_haar_feat2), "M_IOU",
#                     ].values,
#                     alternative="greater",
#                 ).pvalue

#     # plot the wilcoxon results
#     plt.figure()
#     plt.imshow(wilcoxon_results_mat, cmap=cc.cm.fire)
#     plt.colorbar()
#     plt.xticks(np.arange(len(scale_haar_feat_combinations)), scale_haar_feat_combinations)
#     plt.yticks(np.arange(len(scale_haar_feat_combinations)), scale_haar_feat_combinations)
#     plt.xlabel("Scale, n_haar_features")
#     plt.ylabel("Scale, n_haar_features")
#     plt.title("Wilcoxon signed rank test p-values")

x = 1
