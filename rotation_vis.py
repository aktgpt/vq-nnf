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
import itertools
from matplotlib.lines import Line2D
from cycler import cycler

markers = list(Line2D.filled_markers)



sns.set_style("whitegrid")
sns.set_context("talk")  # paper, notebook, talk, poster

plt.rcParams["axes.prop_cycle"] = plt.cycler(linestyle=["-", "-", "-", "-", "-", "-", "-", "-"])
plt.rcParams["font.family"] = "Arial"


folder = "exps_final"

datasets = ["bbs25", "bbs50", "bbs100", "tiny_tlp"]
methods = ["DDIS", "DIWU"]
comp_df = []
for method in methods:
    for dataset in datasets:
        if "bbs" in dataset:
            df = pd.read_csv(f"{folder}/comp/bbs/BBS_rot_{method}_results.csv")
            df_deg0 = pd.read_csv(f"{folder}/comp/bbs/BBS_{method}_27_results.csv")
            df_deg0["rotation"] = 0
            df = pd.concat([df, df_deg0]).reset_index(drop=True)
            df["Method"] = method
            df["exp_cat"] = "BBS" + df["exp_cat"].astype(str)
        else:
            df = pd.read_csv(f"{folder}/comp/tinytlp/TinyTLP_rot_{method}_results.csv")
            df["Method"] = method
            df["exp_cat"] = dataset
            df["iters"] = 1
            df["exp_cat"] = "TinyTLP"
        comp_df.append(df)
comp_df = pd.concat(comp_df).reset_index(drop=True)
comp_df = comp_df.rename(
    columns={
        "exp_cat": "Dataset",
        "mious": "MIoU",
        "sr": "Success Rate",
        "avg_time": "Time",
        "rotation": "Rotation",
    }
)

result_csvs = []
for dataset in datasets:
    result_csv = pd.concat(
        [
            pd.read_csv(x)
            for x in glob.glob(
                f"{folder}/{dataset}/rot/{dataset}*_dataset_all_time_results_k3s2haar_rotation.csv"
            )
        ]
    ).reset_index(drop=True)
    result_csv["Dataset"] = dataset.upper() if "tiny" not in dataset else "TinyTLP"
    result_csv["n_haar_features"] = result_csv["n_haar_features"].apply(
        lambda x: "No Haar" if x == 1 else x
    )
    result_csv["n_haar_features"] = result_csv["n_haar_features"].apply(
        lambda x: f"Haar {x} Rect." if x in [2, 3] else x
    )
    result_csv["n_haar_features"] = result_csv["n_haar_features"].apply(
        lambda x: f"Haar 2 Rect. + 3 Rect." if x == 23 else x
    )
    result_csv["Method"] = (
        "S=" + result_csv["scales"].astype(str) + "," + result_csv["n_haar_features"].astype(str)
    )

    result_csv["time"] = result_csv["Temp_Match_Time"] + result_csv["Kmeans_Time"]
    result_csvs.append(result_csv)


result_csvs = pd.concat(result_csvs).reset_index(drop=True)
result_csvs = result_csvs.rename(
    columns={
        "rotation": "Rotation",
        "M_IOU": "MIoU",
        "Success_Rate": "Success Rate",
        "time": "Time",
    }
)
result_csvs = result_csvs[["Dataset", "Rotation", "Method", "MIoU", "Success Rate", "Time"]]

all_df = pd.concat([comp_df, result_csvs]).reset_index(drop=True)


unique_datasets = all_df["Dataset"].unique()
for dataset in unique_datasets:
    dataset_df = all_df[all_df["Dataset"] == dataset]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    # ax.set_prop_cycle(linestyle=cycler(linestyle=["-", "-", "-", "-", "-", "-", "-", "-"]))
    ax = sns.lineplot(
        data=dataset_df,
        x="Rotation",
        y="MIoU",
        # hue="Method",
        hue="Method",
        # style="Method",
        # kind="line",
        errorbar=None,
        palette="colorblind",
        alpha=1.0,
        ax=ax,
        marker="o",  # markers[: len(dataset_df["Method"].unique())],
        # facet_kws=dict(legend_out=False),
        # ax_=dict(linestyle="-"),
    )

    plt.xticks([0, 60, 120, 180])
    plt.xlabel("Rotation (degrees)")
    plt.title(dataset)
    # put legend at the bottom of the plot
    # g.legend_(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=3)

    plt.savefig(f"exps_final/figures/rot/rot_comp_{dataset}.pdf", dpi=500, bbox_inches="tight")
    plt.close()
x = 1
# ddis_paths = [
#     "exps/comparison/BBS_DDIS_results.csv",
#     "exps/comparison/BBS_rot_DDIS_results.csv",
#     "exps/comparison/TinyTLP_rot_DDIS_results.csv",
# ]
# ddis_df = pd.concat([pd.read_csv(path) for path in ddis_paths]).reset_index(drop=True)
# ddis_df["iters"] = ddis_df["iters"].fillna(1)
# ddis_df["exp_cat"] = ddis_df["exp_cat"].fillna(0)
# ddis_df["rotation"] = ddis_df["rotation"].replace(np.nan, 0).astype(int).astype(str)
# ddis_df = ddis_df.rename(
#     columns={
#         "exp_cat": "Dataset",
#         "mious": "MIoU",
#         "sr": "Success Rate",
#         "avg_time": "Time",
#         "rotation": "Rotation",
#     }
# )
# ddis_df["Method"] = "DDIS"
# # ddis_df["Dataset"] = "BBS" + ddis_df["Dataset"].astype(str)

# diwu_paths = [
#     "exps/comparison/BBS_DIWU_results.csv",
#     "exps/comparison/BBS_rot_DIWU_results.csv",
#     "exps/comparison/TinyTLP_rot_DIWU_results.csv",
# ]
# diwu_df = pd.concat([pd.read_csv(path) for path in diwu_paths])
# diwu_df["iters"] = diwu_df["iters"].fillna(1)
# diwu_df["exp_cat"] = diwu_df["exp_cat"].fillna(0)
# diwu_df["rotation"] = diwu_df["rotation"].replace(np.nan, 0).astype(int).astype(str)
# diwu_df = diwu_df.rename(
#     columns={
#         "exp_cat": "Dataset",
#         "mious": "MIoU",
#         "sr": "Success Rate",
#         "avg_time": "Time",
#         "rotation": "Rotation",
#     }
# )
# diwu_df["Method"] = "DIWU"

# comp_df = pd.concat([ddis_df, diwu_df]).reset_index(drop=True)
# # comp_df = (
# #     comp_df.groupby(["Dataset", "Rotation", "Method"]).mean().reset_index()
# # )
# comp_df["Rotation"] = comp_df["Rotation"].astype(str) + "°"


# bbs_datasets = ["bbs25", "bbs50", "bbs100", "tiny_tlp"]
# n_haar_features = [1, 3, 5, 6]
# scales = [3]
# df = pd.DataFrame()
# for bbs_dataset in bbs_datasets:
#     if "tiny" in bbs_dataset:
#         result_csv = pd.read_csv(
#             "exps/tiny_tlp/tiny_tlp_dataset_all_time_results_optim_rotation.csv"
#         )
#         result_csv["Dataset"] = 0
#     else:
#         result_csv = pd.concat(
#             [
#                 pd.concat(
#                     [
#                         pd.read_csv(
#                             f"exps/{bbs_dataset}/{bbs_dataset}_iter{i}_dataset_all_time_results_optim_rotation.csv"
#                         ),
#                     ]
#                 )
#                 for i in range(1, 4)
#             ]
#         )
#         result_csv = result_csv[
#             (result_csv["n_haar_features"].isin(n_haar_features))
#             & (result_csv["scales"].isin(scales))
#         ]
#         result_csv["Dataset"] = int(bbs_dataset.replace("bbs", ""))

#     result_csv = result_csv.rename(
#         columns={
#             "n_haar_features": "Haar Features",
#             "n_clusters": "Clusters",
#             "n_features": "Features",
#             "scales": "Scales",
#             "M_IOU": "MIoU",
#             "Success_Rate": "Success Rate",
#             "Temp_Match_Time": "Time",
#             "rotation": "Rotation",
#         }
#     )
#     result_csv["Method"] = "Ours"
#     df = pd.concat([df, result_csv]).reset_index(drop=True)

# # result_csv_mean = (
# #     df.groupby(["Features", "Clusters", "Scales", "Haar Features", "Rotation", "Dataset"]).mean().reset_index()
# # )
# df["Rotation"] = df["Rotation"].astype(str) + "°"
# for scale in [3, 4]:
#     df["Method"].loc[np.where((df["Scales"] == scale))] = f"Ours ($s$:{scale}"
# for n_haar_feature in n_haar_features:
#     df["Method"].loc[np.where((df["Haar Features"] == n_haar_feature))] = (
#         df["Method"].iloc[np.where((df["Haar Features"] == n_haar_feature))]
#         + f", $h$:{n_haar_feature})"
#     )

# comp_df = pd.concat(
#     [comp_df, df[["Dataset", "MIoU", "Success Rate", "Time", "Rotation", "Method"]]]
# ).reset_index(drop=True)

# for dataset in [0, 25, 50, 100]:
#     plot_df = comp_df[comp_df["Dataset"] == dataset]
#     if dataset == 0:
#         plot_df["Dataset"] = "TinyTLP"
#     else:
#         plot_df["Dataset"] = "BBS" + plot_df["Dataset"].astype(int).astype(str)
#     # _, ax = plt.subplots(figsize=(10, 6))
#     g = sns.relplot(
#         data=plot_df,
#         x="Rotation",
#         y="MIoU",
#         # hue="Method",
#         hue="Method",
#         # style="Haar Features",
#         # col="Dataset",
#         col="Dataset",
#         kind="line",
#         palette="dark",
#         alpha=0.9,
#         # ax=ax,
#     )
#     plt.savefig(f"exps/comparison/rot_comp_{dataset}.pdf", dpi=500)
#     # plt.close()

# mean_overall_df = comp_df.groupby(["Dataset", "Method"]).mean().reset_index()

# plt.show()
# x = 1

# # result_csv_mean["Rotation"] = result_csv_mean["Rotation"].astype(str)

# # for dataset in [25, 50, 100]:
# #     plot_df = result_csv_mean[result_csv_mean["Dataset"] == dataset]
# #     g = sns.relplot(
# #         data=plot_df,
# #         x="Rotation",
# #         y="MIoU",
# #         # hue="Method",
# #         hue="Haar Features",
# #         # style="Haar Features",
# #         # col="Dataset",
# #         col="Scales",
# #         kind="line",
# #         palette="dark",
# #         alpha=0.8,
# #     )
# #     # plot comp_df on top of df in each subplot
# #     for scale, ax in g.axes_dict.items():
# #         plot_df = comp_df[comp_df["Dataset"] == dataset]
# #         sns.lineplot(
# #             data=plot_df, x="Rotation", y="MIoU", hue="Method", ax=ax, palette="crest", alpha=0.8,
# #         )


# # result_csv_mean.to_csv("exps/bbs25/bbs25_rotation_mean.csv", index=False)
# # divide the value in each feature, cluster, scale, and haar feature by the value in the first rotation
# # to get the relative change in performance in each category
# # for feature in n_features:
# #     for cluster in n_clusters:
# #         for scale in scales:
# #             for haar_feature in n_haar_features:
# #                 result_csv_mean.loc[
# #                     (result_csv_mean["Features"] == feature)
# #                     & (result_csv_mean["Clusters"] == cluster)
# #                     & (result_csv_mean["Scales"] == scale)
# #                     & (result_csv_mean["Haar Features"] == haar_feature),
# #                     "MIoU",
# #                 ] = (
# #                     result_csv_mean["MIoU"][
# #                         (result_csv_mean["Features"] == feature)
# #                         & (result_csv_mean["Clusters"] == cluster)
# #                         & (result_csv_mean["Scales"] == scale)
# #                         & (result_csv_mean["Haar Features"] == haar_feature)
# #                     ]
# #                     / result_csv_mean["MIoU"][
# #                         (result_csv_mean["Features"] == feature)
# #                         & (result_csv_mean["Clusters"] == cluster)
# #                         & (result_csv_mean["Scales"] == scale)
# #                         & (result_csv_mean["Haar Features"] == haar_feature)
# #                         & (result_csv_mean["Rotation"] == 0)
# #                     ].values
# #                 )
# #                 result_csv_mean.loc[
# #                     (result_csv_mean["Features"] == feature)
# #                     & (result_csv_mean["Clusters"] == cluster)
# #                     & (result_csv_mean["Scales"] == scale)
# #                     & (result_csv_mean["Haar Features"] == haar_feature),
# #                     "Success Rate",
# #                 ] = (
# #                     result_csv_mean["Success Rate"][
# #                         (result_csv_mean["Features"] == feature)
# #                         & (result_csv_mean["Clusters"] == cluster)
# #                         & (result_csv_mean["Scales"] == scale)
# #                         & (result_csv_mean["Haar Features"] == haar_feature)
# #                     ]
# #                     / result_csv_mean["Success Rate"][
# #                         (result_csv_mean["Features"] == feature)
# #                         & (result_csv_mean["Clusters"] == cluster)
# #                         & (result_csv_mean["Scales"] == scale)
# #                         & (result_csv_mean["Haar Features"] == haar_feature)
# #                         & (result_csv_mean["Rotation"] == 0)
# #                     ].values
# #                 )

# sns.set_palette(cc.glasbey_light)
# sns.set_style("whitegrid")
# sns.set_context("notebook")  # paper, notebook, talk, poster

# sns.relplot(
#     x="Rotation",
#     y="MIoU",
#     row="Clusters",
#     col="Scales",
#     hue="Haar Features",
#     palette="dark",
#     data=comp_df,
#     kind="line",
#     linewidth=2,
# )
# plt.show()
# x = 1
