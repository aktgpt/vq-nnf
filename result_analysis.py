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
from scipy import ndimage


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


plt.rcParams["font.family"] = "Arial"
# sns.set_context("talk")  # paper, notebook, talk, poster
sns.set(font_scale=2)
sns.set_style("whitegrid")


# result_csv = pd.read_csv("exps_final/tiny_tlp/tiny_tlp_dataset_results.csv")
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
#     palette="colorblind",
#     alpha=0.8,
#     # kind="line",
# )
# g.set(xscale="log")
# plt.show()

bbs_datasets = ["bbs25", "bbs50", "bbs100"]  #

n_features = [27, 512]  # , 64, 128, 256
n_clusters = [4, 8, 16, 32, 64, 128]
n_haar_features = [1, 2, 3, 23]
scales = [1, 2, 3]
result_csvs = []

comp_df = []
for n_feature in [27, 64, 128, 256, 512]:
    ddis_df = pd.read_csv(f"exps_final/comp/bbs/BBS_DDIS_{n_feature}_results.csv")
    ddis_df["Method"] = "DDIS"
    ddis_df["Feature Dimensions"] = n_feature
    ddis_df = ddis_df.rename(columns={"exp_cat": "Dataset"})

    diwu_df = pd.read_csv(f"exps_final/comp/bbs/BBS_DIWU_{n_feature}_results.csv")
    diwu_df["Method"] = "DIWU"
    diwu_df["Feature Dimensions"] = n_feature
    diwu_df = diwu_df.rename(columns={"exp_cat": "Dataset"})

    comp_df.append(ddis_df)
    comp_df.append(diwu_df)

comp_df = pd.concat(comp_df).reset_index(drop=True)
comp_df.drop(columns=["iters"], inplace=True)
comp_df_mean_dataset = (
    comp_df.groupby(["Dataset", "Method", "Feature Dimensions"]).mean().reset_index()
)
comp_df_mean_dataset.to_csv("exps_final/comp/bbs/bbs_comp.csv", index=False)
comp_df_all_dataset = comp_df.groupby(["Method", "Feature Dimensions"]).mean().reset_index()
comp_df_all_dataset.to_csv("exps_final/comp/bbs/bbs_comp_all_dataset.csv", index=False)

for bbs_dataset in bbs_datasets:
    result_csv = pd.concat(
        [
            pd.read_csv(x)
            for x in glob.glob(
                f"exps_final/{bbs_dataset}/{bbs_dataset}_iter*_dataset_all_time_results_k3s2haar.csv"
            )
        ]
        # [
        #     pd.concat(
        #         [
        #             pd.read_csv(
        #                 f"exps_final/{bbs_dataset}/{bbs_dataset}_iter{i}_dataset_all_time_results_k3s2haar.csv"
        #             ),
        #             # pd.read_csv(
        #             #     f"exps_final/{bbs_dataset}/{bbs_dataset}_iter{i}_dataset_all_time_results_optim_27.csv"
        #             # ),
        #             # pd.read_csv(
        #             #     f"exps_final/{bbs_dataset}/{bbs_dataset}_iter{i}_dataset_all_time_results_optim_haar6.csv"
        #             # ),
        #         ]
        #     )
        #     for i in range(1, 6)
        # ]
    )
    result_csv = result_csv.rename(
        columns={
            "n_haar_features": "Haar Features",
            "n_clusters": "Clusters",
            "n_features": "Feature Dimensions",
            "scales": "Scales",
            "M_IOU": "MIoU",
            "Success_Rate": "Success Rate",
        }
    )
    result_csv["Total Time (sec.)"] = result_csv["Temp_Match_Time"] + result_csv["Kmeans_Time"]
    result_csv["Dataset"] = int(bbs_dataset.replace("bbs", ""))

    # result_csvs.append(result_csv)

    # result_csvs = pd.concat(result_csvs)

    result_csv_mean = (
        result_csv.groupby(["Feature Dimensions", "Clusters", "Scales", "Haar Features", "Dataset"])
        .mean()
        .reset_index()
    )
    result_csv_mean["stdMIoU"] = (
        result_csv.groupby(["Feature Dimensions", "Clusters", "Scales", "Haar Features", "Dataset"])
        .std()
        .reset_index()["MIoU"]
    )
    result_csv_mean["stdSR"] = (
        result_csv.groupby(["Feature Dimensions", "Clusters", "Scales", "Haar Features", "Dataset"])
        .std()
        .reset_index()["Success Rate"]
    )
    result_csv_mean.to_csv(f"exps_final/{bbs_dataset}/{bbs_dataset}_mean_optim.csv", index=False)

    result_csv_plot = result_csv_mean[
        (result_csv_mean["Feature Dimensions"].isin(n_features))
        & (result_csv_mean["Clusters"].isin(n_clusters))
        & (result_csv_mean["Haar Features"].isin(n_haar_features))
        & (result_csv_mean["Scales"].isin(scales))
    ]

    # best = (
    #     result_csv_mean.groupby(["Feature Dimensions"])
    #     .apply(lambda x: x.nlargest(1, columns=["MIoU"]))
    #     .reset_index(level=0, drop=True)
    # ).reset_index(drop=True)
    # print(best)

    #  shift the Clusters of scales 2 and 4 by a little for plotting
    result_csv_plot.loc[result_csv_plot["Scales"] == 1, "Clusters"] = result_csv_plot.loc[
        result_csv_plot["Scales"] == 1, "Clusters"
    ] * (1 - 0.15)
    result_csv_plot.loc[result_csv_plot["Scales"] == 2, "Clusters"] = result_csv_plot.loc[
        result_csv_plot["Scales"] == 2, "Clusters"
    ] * (1 - 0.0)
    result_csv_plot.loc[result_csv_plot["Scales"] == 3, "Clusters"] = result_csv_plot.loc[
        result_csv_plot["Scales"] == 3, "Clusters"
    ] * (1 + 0.15)
    # result_csv_plot.loc[result_csv_plot["Scales"] == 4, "Clusters"] = result_csv_plot.loc[
    #     result_csv_plot["Scales"] == 4, "Clusters"
    # ] * (1 + 0.2)
    result_csv_plot["Haar Features"] = result_csv_plot["Haar Features"].apply(
        lambda x: "No Haar" if x == 1 else x
    )
    result_csv_plot["Haar Features"] = result_csv_plot["Haar Features"].apply(
        lambda x: f"{x} Rect." if x in [2, 3] else x
    )
    result_csv_plot["Haar Features"] = result_csv_plot["Haar Features"].apply(
        lambda x: f"2 Rect. + 3 Rect." if x == 23 else x
    )

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    g = sns.relplot(
        data=result_csv_plot,
        y="MIoU",
        x="Clusters",
        # x="Clusters",
        # x="Complexity",
        hue="Scales",
        style="Haar Features",
        row="Feature Dimensions",
        # row="Scales",
        palette="colorblind",
        alpha=0.9,
        s=200,
        height=5,
        aspect=1.4,
        legend=False,
        # linewidth=1,
        # edgecolors="face"
        # kind="line",
        # ax=ax
    )
    plt.xscale("log", base=2)

    ## draw a horizontal line at at each plot in the column with values of DDIS and DIWU
    for ii, ax in enumerate(g.axes.flat):
        ax.axhline(
            y=comp_df.loc[
                (comp_df["Dataset"] == int(bbs_dataset.replace("bbs", "")))
                & (comp_df["Feature Dimensions"] == n_features[ii])
                & (comp_df["Method"] == "DDIS")
            ]["mious"].values[0],
            color=sns.color_palette("colorblind")[3],
            linestyle="--",
            label="DDIS",
            linewidth=3,
        )
        ax.axhline(
            y=comp_df.loc[
                (comp_df["Dataset"] == int(bbs_dataset.replace("bbs", "")))
                & (comp_df["Feature Dimensions"] == n_features[ii])
                & (comp_df["Method"] == "DIWU")
            ]["mious"].values[0],
            color=sns.color_palette("colorblind")[4],
            linestyle="--",
            label="DIWU",
            linewidth=3,
        )
    # plt.legend()
    plt.savefig(f"exps_final/figures/bbs/{bbs_dataset}_optim_mean_miou.pdf", dpi=500)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    g = sns.relplot(
        data=result_csv_plot,
        y="Success Rate",
        x="Clusters",
        # x="Clusters",
        # x="Complexity",
        hue="Scales",
        style="Haar Features",
        row="Feature Dimensions",
        # row="Scales",
        palette="colorblind",
        alpha=0.9,
        s=200,
        height=5,
        aspect=1.4,
        legend=False,
        # linewidth=1,
        # edgecolors="face"
        # kind="line",
        # ax=ax
    )
    for ii, ax in enumerate(g.axes.flat):
        ax.axhline(
            y=comp_df.loc[
                (comp_df["Dataset"] == int(bbs_dataset.replace("bbs", "")))
                & (comp_df["Feature Dimensions"] == n_features[ii])
                & (comp_df["Method"] == "DDIS")
            ]["sr"].values[0],
            color=sns.color_palette("colorblind")[3],
            linestyle="--",
            label="DDIS",
            linewidth=3,
        )
        ax.axhline(
            y=comp_df.loc[
                (comp_df["Dataset"] == int(bbs_dataset.replace("bbs", "")))
                & (comp_df["Feature Dimensions"] == n_features[ii])
                & (comp_df["Method"] == "DIWU")
            ]["sr"].values[0],
            color=sns.color_palette("colorblind")[4],
            linestyle="--",
            label="DIWU",
            linewidth=3,
        )
    # plt.legend()
    plt.xscale("log", base=2)
    plt.savefig(f"exps_final/figures/bbs/{bbs_dataset}_optim_mean_sr.pdf", dpi=500)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    g = sns.relplot(
        data=result_csv_plot,
        y="Total Time (sec.)",
        x="Clusters",
        # x="Clusters",
        # x="Complexity",
        hue="Scales",
        style="Haar Features",
        row="Feature Dimensions",
        # row="Scales",
        palette="colorblind",
        alpha=0.9,
        s=200,
        height=5,
        aspect=1.4,
        legend=False,
        # kind="line",
        # ax=ax
    )
    # for ii, ax in enumerate(g.axes.flat):
    #     ax.axhline(
    #         y=comp_df.loc[
    #             (comp_df["Dataset"] == int(bbs_dataset.replace("bbs", "")))
    #             & (comp_df["Feature Dimensions"] == n_features[ii])
    #             & (comp_df["Method"] == "DDIS")
    #         ]["avg_time"].values[0],
    #         color=sns.color_palette("colorblind")[3],
    #         linestyle="--",
    #         label="DDIS",
    #     )
    #     ax.axhline(
    #         y=comp_df.loc[
    #             (comp_df["Dataset"] == int(bbs_dataset.replace("bbs", "")))
    #             & (comp_df["Feature Dimensions"] == n_features[ii])
    #             & (comp_df["Method"] == "DIWU")
    #         ]["avg_time"].values[0],
    #         color=sns.color_palette("colorblind")[4],
    #         linestyle="--",
    #         label="DIWU",
    #     )
    # plt.legend()
    plt.xscale("log", base=2)
    # plt.yscale("log", base=10)
    plt.savefig(f"exps_final/figures/bbs/{bbs_dataset}_optim_mean_time.pdf", dpi=500)
    result_csvs.append(result_csv)

result_csvs = pd.concat(result_csvs).reset_index(drop=True)


result_csv_mean_all_datasets = (
    result_csvs.groupby(["Feature Dimensions", "Clusters", "Scales", "Haar Features"])
    .mean()
    .reset_index()
)
result_csv_mean_all_datasets.to_csv(f"exps_final/stats/bbs/mean_all_datasets.csv", index=False)
plt.show()

# for dataset in result_csv_mean_dataset["Dataset"].unique():
#     all_features = result_csv_mean_dataset["Features"].unique()
#     fig, ax = plt.subplots(1, len(all_features), figsize=(20, 5))
#     dataset_df = result_csv_mean_dataset.loc[(result_csv_mean_dataset["Dataset"] == dataset)]
#     vmin = dataset_df["MIoU"].min()
#     vmax = dataset_df["MIoU"].max()
#     for i, feature in enumerate(all_features):
#         df_test = dataset_df.loc[
#             (result_csv_mean_dataset["Features"] == feature)
#             & (result_csv_mean_dataset["Clusters"] == 128)
#         ].reset_index(drop=True)
#         x = df_test.pivot(index="Scales", columns="Haar Features", values="MIoU")
#         sns.heatmap(
#             x,
#             annot=True,
#             ax=ax[all_features.tolist().index(feature)],
#             vmin=vmin,
#             vmax=vmax,
#             cbar=False,
#         )
#         ax[all_features.tolist().index(feature)].set_title(f"Features {feature}")
#     fig.colorbar(ax[-2].collections[0], ax=ax[-1])
#     plt.suptitle(f"Dataset {dataset}")

# plt.show()

best = (
    result_csv_mean_dataset.groupby(["Dataset", "Features"])
    .apply(lambda x: x.nlargest(2, columns=["MIoU"]))
    .reset_index(level=0, drop=True)
).reset_index(drop=True)
best["Dataset"] = "BBS" + best["Dataset"].astype(str)
# best.groupby(["Features", "Scales", "Haar Features"]).size()
best.to_csv("exps_final/bbs_optim_best.csv", index=False)

# best_each_dataset = result_csv_mean_dataset.loc[
#     result_csv_mean_dataset["Features"].isin(best["Features"].values)
#     & result_csv_mean_dataset["Clusters"].isin(best["Clusters"].values)
#     & result_csv_mean_dataset["Scales"].isin(best["Scales"].values)
#     & result_csv_mean_dataset["Haar Features"].isin(best["Haar Features"].values)
# ].reset_index(drop=True)

# best.pivot(columns="Features", index="Dataset", values="Scales").plot.bar()
# best.pivot(columns="Features", index="Dataset", values="Haar Features").plot.bar()
# best.pivot(
#     columns=["Features", "Scales", "Haar Features"], index="Dataset", values="MIoU"
# ).plot.bar()
# sns.barplot(data=best, x="Dataset", y="MIoU", hue="Features")

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
sns.barplot(data=best, hue="Features", y="MIoU", x="Dataset", ax=ax1)
ax1.axhline(0, color="k", clip_on=False)
ax1.set_ylabel("MIoU")

sns.barplot(data=best, hue="Features", y="Scales", x="Dataset", ax=ax2)
ax2.axhline(0, color="k", clip_on=False)
ax2.set_ylabel("Scales")

sns.barplot(data=best, hue="Features", y="Haar Features", x="Dataset", ax=ax3)
ax3.axhline(0, color="k", clip_on=False)
ax3.set_ylabel("Haar Features")

sns.barplot(data=best, hue="Features", y="Total Time (sec.)", x="Dataset", ax=ax4)
ax4.axhline(0, color="k", clip_on=False)
ax4.set_ylabel("Total Time (sec.)")

sns.despine(bottom=True)
# plt.setp(f.axes, yticks=[])
# plt.tight_layout(h_pad=1)

plt.show()
x = 1

result_csv_mean = (
    result_csvs.groupby(["Features", "Clusters", "Scales", "Haar Features"]).mean().reset_index()
)
best = (
    result_csv_mean.groupby(["Features"])
    .apply(lambda x: x.nlargest(1, columns=["MIoU"]))
    .reset_index(level=0, drop=True)
).reset_index(drop=True)

result_csv_mean_dataset = (
    result_csvs.groupby(["Features", "Clusters", "Scales", "Haar Features", "Dataset"])
    .mean()
    .reset_index()
)


best_each_dataset = result_csv_mean_dataset.loc[
    result_csv_mean_dataset["Features"].isin(best["Features"].values)
    & result_csv_mean_dataset["Clusters"].isin(best["Clusters"].values)
    & result_csv_mean_dataset["Scales"].isin(best["Scales"].values)
    & result_csv_mean_dataset["Haar Features"].isin(best["Haar Features"].values)
].reset_index(drop=True)


result_csv_std = (
    result_csvs.groupby(["Features", "Clusters", "Scales", "Haar Features", "Dataset"])
    .std()
    .reset_index()
)

x = result_csv_mean.loc[
    (result_csv_mean["Features"] == 256)
    & (result_csv_mean["Clusters"] == 64)
    & (result_csv_mean["Scales"] == 2)
    & (result_csv_mean["Haar Features"] == 5)
]
x1 = result_csv_std.loc[
    (result_csv_std["Features"] == 256)
    & (result_csv_std["Clusters"] == 64)
    & (result_csv_std["Scales"] == 2)
    & (result_csv_std["Haar Features"] == 5)
]
x2 = x.groupby(["Features", "Clusters", "Scales", "Haar Features"]).mean().reset_index()


result_csv_mean.to_csv(f"exps_final/{bbs_dataset}/{bbs_dataset}_mean_optim.csv", index=False)

result_csv_std.to_csv(f"exps_final/{bbs_dataset}/{bbs_dataset}_std_optim.csv", index=False)

top3 = (
    result_csv_mean.groupby(["Features"], sort=False)
    .apply(lambda x: x.nlargest(5, columns=["MIoU"]))
    .reset_index(level=0, drop=True)
).reset_index(drop=True)
print(top3)


# sns.move_legend(
#     g, loc="lower center", ncol=8, title=None, bbox_to_anchor=(0.5, -0.05),
# )

# plt.savefig("exps_final/bbs100/bbs100.pdf", dpi=500)

