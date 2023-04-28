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


def round_off_rating(number, base=0.5):
    """Round a number to the closest half integer.
    >>> round_off_rating(1.3)
    1.5
    >>> round_off_rating(2.6)
    2.5
    >>> round_off_rating(3.0)
    3.0
    >>> round_off_rating(4.1)
    4.0"""

    return round(number * (1 / base)) / (1 / base)


sns.set_style("whitegrid")
sns.set_context("talk")  # paper, notebook, talk, poster

datasets = ["bbs25", "bbs50", "bbs100", "tlpattr", "tiny_tlp"]  # , "bbs50", "bbs100"
n_features = [27, 512]

for dataset in datasets:
    all_result_csv = pd.concat(
        pd.read_csv(file_name)
        for file_name in glob.glob(
            f"exps_final/{dataset}/gauss/{dataset}*_dataset_all_time_results_k3s2haar_gauss.csv"
        )
    )
    all_result_csv = (
        all_result_csv.groupby(
            ["n_features", "n_clusters", "n_haar_features", "kernel_size", "sigma"]
        )
        .mean()
        .reset_index()
    )
    all_result_csv["Method"] = (
        all_result_csv["kernel_size"].astype(str) + "x" + all_result_csv["sigma"].astype(str)
    )

    all_result_csv.to_csv(f"exps_final/stats/gauss/{dataset}_gauss.csv")

    for n_feature in n_features:
        # result_csv = pd.read_csv(
        #     f"exps_final/tlpattr/kernel_sigma/tlpattr_dataset_all_time_results_k3s2haar_gauss{n_feature}.csv"
        # )
        # result_csv = pd.concat(
        #     pd.read_csv(file_name)
        #     for file_name in glob.glob(
        #         f"exps_final/bbs100/gauss/bbs100_iter*_dataset_all_time_results_k3s2haar_gauss{n_feature}.csv"
        #     )
        # )
        result_csv = all_result_csv[all_result_csv["n_features"] == n_feature]

        result_csv["n_haar_features"] = result_csv["n_haar_features"].astype(str)
        result_csv["n_haar_features"] = pd.Categorical(
            result_csv["n_haar_features"], ["1", "2", "3", "23"]
        )
        result_csv = result_csv.rename(
            columns={
                "n_haar_features": "Haar Filters",
                "sigma": "Sigma",
                "kernel_size": "Kernel Size",
                "M_IOU": "MIoU",
                "Success_Rate": "Success Rate",
            }
        )
        result_csv["Time"] = result_csv["Temp_Match_Time"] + result_csv["Kmeans_Time"]
        g = sns.relplot(
            data=result_csv,
            y="MIoU",
            x="Haar Filters",
            hue="Method",  # "Sigma",
            # col="Kernel Size",
            col="n_clusters",
            palette="dark",
            alpha=0.8,
            kind="line",
            linewidth=2,
        )
        # for ii, ax in enumerate(g.axes.flat):
        #     ax.axhline(
        #         y=0.505 if n_feature == 27 else 0.549,
        #         color="black",
        #         linestyle="--",
        #         label="Best 3x3 Results",
        #     )
        plt.legend()
        plt.savefig(f"exps_final/figures/gauss/{dataset}_{n_feature}_gauss_mious.pdf", dpi=500)
        g = sns.relplot(
            data=result_csv,
            y="Time",
            x="Haar Filters",
            hue="Method",  # "Sigma",
            # col="Kernel Size",
            col="n_clusters",
            palette="dark",
            alpha=0.8,
            kind="line",
            linewidth=2,
        )
        # for ii, ax in enumerate(g.axes.flat):
        #     ax.axhline(
        #         y=1.159 if n_feature == 27 else 0.911,
        #         color="black",
        #         linestyle="--",
        #         label="Best 3x3 Results",
        #     )
        plt.legend()
        plt.savefig(f"exps_final/figures/gauss/{dataset}_{n_feature}_gauss_time.pdf", dpi=500)


# plt.show()
x = 1


# result_csv = pd.read_csv("exps/tiny_tlp/tiny_tlp_dataset_results.csv")
# g = sns.relplot(
#     data=result_csv,
#     # y="Success_Rate",
#     y="M_IOU",
#     x="n_clusters",
#     # x="Complexity",
#     hue="scales",
#     style="n_haar_features",
#     col="n_features",
#     row="scales",
#     palette="dark",
#     alpha=0.8,
#     # kind="line",
# )
# g.set(xscale="log")
# plt.show()
datasets = ["bbs25", "bbs50", "bbs100"]
kernel_sizes = [3]
n_features = [64, 128, 256, 512]
n_clusters = [4, 8, 16, 32, 64, 128]
n_haar_features = [1, 3, 5, 6]
scales = [1, 2, 3, 4]
sigtest_csv = []

for dataset in datasets:
    result_csvs = []
    for kernel_size in kernel_sizes:
        try:
            result_csv = pd.concat(
                [
                    pd.concat(
                        [
                            pd.read_csv(file_name)
                            for file_name in glob.glob(
                                f"exps/{dataset}/gauss/{dataset}_iter*_dataset_all_time_results_k{kernel_size}s2haar.csv"
                            )
                        ]
                    )
                ]
            )

            result_csv = result_csv.rename(
                columns={
                    "n_haar_features": "Haar Features",
                    "n_clusters": "Clusters",
                    "n_features": "Features",
                    "scales": "Scales",
                    "M_IOU": "MIoU",
                    "Success_Rate": "Success Rate",
                    "Temp_Match_Time": "Time",
                }
            )
            result_csv["Kernel Size"] = kernel_size
            result_csvs.append(result_csv)
        except:
            continue
    result_csv = pd.concat(result_csvs)
    df_mean = (
        result_csv.groupby(["Kernel Size", "Features", "Clusters", "Scales", "Haar Features"])
        .mean()
        .reset_index()
    )
    df_mean.to_csv(f"exps/{dataset}/gauss/{dataset}_dataset_all_time_k3haar.csv", index=False)

    df_mean.loc[df_mean["Haar Features"] == 1, "Clusters"] = df_mean.loc[
        df_mean["Haar Features"] == 1, "Clusters"
    ] * (1 - 0.10)
    df_mean.loc[df_mean["Haar Features"] == 3, "Clusters"] = df_mean.loc[
        df_mean["Haar Features"] == 3, "Clusters"
    ] * (1 - 0.05)
    df_mean.loc[df_mean["Haar Features"] == 5, "Clusters"] = df_mean.loc[
        df_mean["Haar Features"] == 5, "Clusters"
    ] * (1 + 0.05)
    df_mean.loc[df_mean["Haar Features"] == 6, "Clusters"] = df_mean.loc[
        df_mean["Haar Features"] == 6, "Clusters"
    ] * (1 + 0.10)

    g = sns.relplot(
        data=df_mean,
        y="MIoU",
        x="Clusters",
        # x="Clusters",
        hue="Scales",
        style="Haar Features",
        col="Features",
        # col="",
        palette="dark",
        alpha=0.7,
        # s=200,
        # linewidth=1,
        # edgecolors="face"
        # kind="line",
        # ax=ax
    )
    plt.xscale("log", base=2)
# plt.show()
x = 1

kernel_sizes = [3, 5, 7]
sigmas = [4, 3, 2, 1.5]
result_csvs = []
for kernel_size in kernel_sizes:
    for sigma in sigmas:
        try:
            result_csv = pd.concat(
                [
                    pd.concat(
                        [
                            pd.read_csv(file_name)
                            for file_name in glob.glob(
                                f"exps/bbs25/gauss/bbs25_iter*_dataset_all_time_results_k{kernel_size}s{sigma}haar.csv"
                            )
                        ]
                    )
                ]
            )
        except:
            continue

        result_csv = result_csv.rename(
            columns={
                "n_haar_features": "Haar Features",
                "n_clusters": "Clusters",
                "n_features": "Features",
                "scales": "Scales",
                "M_IOU": "MIoU",
                "Success_Rate": "Success Rate",
                "Temp_Match_Time": "Time",
            }
        )
        result_csv["Kernel Size"] = f"{kernel_size}_{sigma}"
        # result_csv["Sigma"] = sigma
        result_csvs.append(result_csv)

result_csv = pd.concat(result_csvs)
df_mean = (
    result_csv.groupby(["Kernel Size", "Features", "Clusters", "Scales", "Haar Features"])
    .mean()
    .reset_index()
)
df_mean.to_csv("exps/bbs25/gauss/bbs25_dataset_all_time_results_haar.csv", index=False)

g = sns.relplot(
    data=result_csv,
    y="MIoU",
    x="Haar Features",
    # x="Clusters",
    hue="Kernel Size",
    style="Features",
    row="Clusters",
    col="Scales",
    palette="dark",
    alpha=0.7,
    # s=200,
    # linewidth=1,
    # edgecolors="face"
    kind="line",
    # ax=ax
)
plt.show()
x = 1
