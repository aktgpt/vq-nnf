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

sns.set_style("whitegrid")
sns.set_context("notebook")  # paper, notebook, talk, poster

n_features = [64, 128, 256, 512]
n_clusters = [4, 8, 16, 32, 64, 128]
n_haar_features = [1, 4, 6]
scales = [1, 2, 3]

result_csv = pd.concat(
    [
        pd.read_csv(f"exps/bbs25/bbs25_iter{i}_dataset_params_explore_2x3x4x.csv")
        for i in range(1, 5)
    ]
)

# result_csv = result_csv[
#     # (result_csv["n_features"].isin(n_features))&
#     (result_csv["n_clusters"].isin(n_clusters))
#     & (result_csv["n_haar_features"].isin(n_haar_features))
#     & (result_csv["scales"].isin(scales))
# ]
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
result_csv_mean = (
    result_csv.groupby(
        [
            "Features",
            "Clusters",
            "Scales",
            "Haar Features",
            "param_1x",
            "param_2x",
            "param_3x",
            "param_4x",
        ]
    )
    .mean()
    .reset_index()
)
result_csv_mean.to_csv("exps/bbs25/bbs25_dataset_params_explore_2x3x4x.csv", index=False)

top3 = (
    result_csv_mean.groupby(["Features", "Clusters", "Scales", "Haar Features"], sort=False)
    .apply(lambda x: x.nlargest(3, columns=["MIoU"]))
    .reset_index(level=0, drop=True)
).reset_index(drop=True)
top3.to_csv("exps/bbs25/bbs25_dataset_params_explore_2x3x4x_top3.csv", index=False)

x = 1


# data_mat = np.zeros()


# g = sns.pairplot(
#     result_csv_mean, hue="MIoU", vars=["param_2x", "param_3x", "param_4x"], palette="light:#5A9",
# )
g = sns.relplot(
    data=result_csv_mean,
    y="MIoU",
    x="param_4x",
    hue="Scales",
    style="Clusters",
    row="param_2x",
    col="param_3x",
    palette="dark",
    alpha=0.6,
    kind="line",
    linewidth=2,
)
plt.xscale("symlog", base=2, linthresh=0.5)

# g = sns.relplot(
#     data=result_csv_mean,
#     y="MIoU",
#     x="param_4x",
#     hue="param_2x",
#     style="Scales",
#     row="param_3x",
#     col="Clusters",
#     palette="dark",
#     alpha=0.6,
#     kind="line",
#     linewidth=2,
# )
# plt.xscale("symlog", base=2, linthresh=0.5)


g = sns.relplot(
    data=result_csv_mean,
    y="MIoU",
    x="param_3x",
    hue="Scales",
    style="Clusters",
    row="param_2x",
    col="param_4x",
    palette="dark",
    alpha=0.6,
    kind="line",
    linewidth=2,
)
plt.xscale("symlog", base=2, linthresh=0.5)

g = sns.relplot(
    data=result_csv_mean,
    y="MIoU",
    x="param_2x",
    hue="Scales",
    style="Clusters",
    row="param_3x",
    col="param_4x",
    palette="dark",
    alpha=0.6,
    kind="line",
    linewidth=2,
)
plt.xscale("symlog", base=2, linthresh=0.5)

# for ax in g.axes.flat:
#     ax.set_xscale("symlog", base=2, linthresh=0.5)
plt.show()

x = 1

# result_csv_mean1 = result_csv_mean.pivot(  # .unstack(),
#     index=["param_1x", "param_2x", "param_3x", "param_4x"],
#     columns=["Clusters", "Scales", "Haar Features"],
#     values="MIoU",
# )
# result_csv_mean1 = result_csv_mean1.reset_index()


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.cm as cm


# def fun(x, y):
#     return result_csv_mean["MIoU"][
#         np.where((result_csv_mean["param_2x"] == x) & (result_csv_mean["param_3x"] == y))
#     ]


# x = result_csv_mean["param_2x"]
# y = result_csv_mean["param_3x"]
# z = result_csv_mean["param_4x"]
# c = result_csv_mean["MIoU"]

# X, Y, Z = np.meshgrid(x, y, z)
# # c = fun(X.ravel(), Y.ravel()).reshape(X.shape)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# r1 = ax.plot_surface(
#     X,Y,Z, cmap="hot"
# )
# fc = r1.get_facecolors()
# ax.clear()
# ax.plot_surface(
#     result_csv_mean["param_2x", "MIoU"],
#     result_csv_mean["param_3x", "MIoU"],
#     result_csv_mean["param_4x", "MIoU"],
#     facecolors=fc,
# )
# ax.set_xlabel("param_2x")
# ax.set_ylabel("param_3x")
# ax.set_zlabel("param_4x")
# plt.show()


# g = sns.relplot(
#     data=result_csv_mean,
#     # y="Success_Rate",
#     y="MIoU",
#     x="param_4x",
#     # x="Clusters",
#     # x="Complexity",
#     hue="Scales",
#     style="Features",
#     col="Clusters",
#     # row="Features",
#     palette="dark",
#     alpha=0.6,
#     # s=200,
#     # linewidth=1,
#     # edgecolors="face"
#     kind="line",
#     # ax=ax
# )
# plt.xscale("symlog", base=2)


# result_csv = pd.concat(
#     [pd.read_csv(f"exps/bbs25/bbs25_iter{i}_dataset_params_explore_4x.csv") for i in range(1, 6)]
# )

# # result_csv = result_csv[
# #     # (result_csv["n_features"].isin(n_features))&
# #     (result_csv["n_clusters"].isin(n_clusters))
# #     & (result_csv["n_haar_features"].isin(n_haar_features))
# #     & (result_csv["scales"].isin(scales))
# # ]
# result_csv = result_csv.rename(
#     columns={
#         "n_haar_features": "Haar Features",
#         "n_clusters": "Clusters",
#         "n_features": "Features",
#         "scales": "Scales",
#         "M_IOU": "MIoU",
#         "Success_Rate": "Success Rate",
#         "Temp_Match_Time": "Time",
#     }
# )
# result_csv_mean = (
#     result_csv.groupby(
#         [
#             "Features",
#             "Clusters",
#             "Scales",
#             "Haar Features",
#             "param_1x",
#             "param_2x",
#             "param_3x",
#             "param_4x",
#         ]
#     )
#     .mean()
#     .reset_index()
# )

# g = sns.relplot(
#     data=result_csv_mean,
#     # y="Success_Rate",
#     y="MIoU",
#     x="param_4x",
#     # x="Clusters",
#     # x="Complexity",
#     hue="Scales",
#     style="Features",
#     col="Clusters",
#     # row="Features",
#     palette="dark",
#     alpha=0.6,
#     # s=200,
#     # linewidth=1,
#     # edgecolors="face"
#     kind="line",
#     # ax=ax
# )
# plt.xscale("symlog", base=2)

# result_csv = pd.concat(
#     [pd.read_csv(f"exps/bbs25/bbs25_iter{i}_dataset_params_explore_3x.csv") for i in range(1, 5)]
# )

# # result_csv = result_csv[
# #     # (result_csv["n_features"].isin(n_features))&
# #     (result_csv["n_clusters"].isin(n_clusters))
# #     & (result_csv["n_haar_features"].isin(n_haar_features))
# #     & (result_csv["scales"].isin(scales))
# # ]
# result_csv = result_csv.rename(
#     columns={
#         "n_haar_features": "Haar Features",
#         "n_clusters": "Clusters",
#         "n_features": "Features",
#         "scales": "Scales",
#         "M_IOU": "MIoU",
#         "Success_Rate": "Success Rate",
#         "Temp_Match_Time": "Time",
#     }
# )
# result_csv_mean = (
#     result_csv.groupby(
#         [
#             "Features",
#             "Clusters",
#             "Scales",
#             "Haar Features",
#             "param_1x",
#             "param_2x",
#             "param_3x",
#             "param_4x",
#         ]
#     )
#     .mean()
#     .reset_index()
# )

# g = sns.relplot(
#     data=result_csv_mean,
#     # y="Success_Rate",
#     y="MIoU",
#     x="param_3x",
#     # x="Clusters",
#     # x="Complexity",
#     hue="Scales",
#     style="Features",
#     col="Clusters",
#     # row="Features",
#     palette="dark",
#     alpha=0.6,
#     # s=200,
#     # linewidth=1,
#     # edgecolors="face"
#     kind="line",
#     # ax=ax
# )
# plt.xscale("symlog", base=2)

# # Draw horizontal lines at the start values for reference on the relplot above
# # for ax in g.axes.flat:
# # ax.axhline(y=)


# result_csv = pd.concat(
#     [pd.read_csv(f"exps/bbs25/bbs25_iter{i}_dataset_params_explore_2x.csv") for i in range(1, 6)]
# )

# # result_csv = result_csv[
# #     # (result_csv["n_features"].isin(n_features))&
# #     (result_csv["n_clusters"].isin(n_clusters))
# #     & (result_csv["n_haar_features"].isin(n_haar_features))
# #     & (result_csv["scales"].isin(scales))
# # ]
# result_csv = result_csv.rename(
#     columns={
#         "n_haar_features": "Haar Features",
#         "n_clusters": "Clusters",
#         "n_features": "Features",
#         "scales": "Scales",
#         "M_IOU": "MIoU",
#         "Success_Rate": "Success Rate",
#         "Temp_Match_Time": "Time",
#     }
# )
# result_csv_mean = (
#     result_csv.groupby(
#         [
#             "Features",
#             "Clusters",
#             "Scales",
#             "Haar Features",
#             "param_1x",
#             "param_2x",
#             "param_3x",
#             "param_4x",
#         ]
#     )
#     .mean()
#     .reset_index()
# )

# g = sns.relplot(
#     data=result_csv_mean,
#     # y="Success_Rate",
#     y="MIoU",
#     x="param_2x",
#     # x="Clusters",
#     # x="Complexity",
#     hue="Scales",
#     style="Features",
#     col="Clusters",
#     # row="Features",
#     palette="dark",
#     alpha=0.6,
#     # s=200,
#     # linewidth=1,
#     # edgecolors="face"
#     kind="line",
#     # ax=ax
# )
# plt.xscale("symlog", base=2)


# result_csv = pd.concat(
#     [pd.read_csv(f"exps/bbs25/bbs25_iter{i}_dataset_params_explore_1x.csv") for i in range(1, 5)]
# )

# result_csv = result_csv[
#     # (result_csv["n_features"].isin(n_features))&
#     (result_csv["n_clusters"].isin(n_clusters))
#     & (result_csv["n_haar_features"].isin(n_haar_features))
#     & (result_csv["scales"].isin(scales))
# ]
# result_csv = result_csv.rename(
#     columns={
#         "n_haar_features": "Haar Features",
#         "n_clusters": "Clusters",
#         "n_features": "Features",
#         "scales": "Scales",
#         "M_IOU": "MIoU",
#         "Success_Rate": "Success Rate",
#         "Temp_Match_Time": "Time",
#     }
# )
# result_csv_mean = (
#     result_csv.groupby(
#         [
#             "Features",
#             "Clusters",
#             "Scales",
#             "Haar Features",
#             "param_1x",
#             "param_2x",
#             "param_3x",
#             "param_4x",
#         ]
#     )
#     .mean()
#     .reset_index()
# )

# g = sns.relplot(
#     data=result_csv_mean,
#     # y="Success_Rate",
#     y="MIoU",
#     x="param_1x",
#     # x="Clusters",
#     # x="Complexity",
#     hue="Scales",
#     style="Features",
#     col="Clusters",
#     # row="Features",
#     palette="dark",
#     alpha=0.6,
#     # s=200,
#     # linewidth=1,
#     # edgecolors="face"
#     kind="line",
#     # ax=ax
# )
# plt.xscale("symlog", base=2)


# plt.show()

# x = 1
