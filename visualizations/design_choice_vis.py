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
import decimal

sns.set_style("whitegrid")
# sns.set_context("notebook")  # paper, notebook, talk, poster
sns.set(font_scale=2.5)
plt.rcParams["font.family"] = "Arial"

n_features = [27, 512]
n_clusters = [4, 8, 16, 32, 64, 128]
n_haar_features = [1, 2, 3, 23]
scales = [1, 2, 3]

datasets = ["BBS25", "BBS50", "BBS100", "TLPattr", "TinyTLP"]
methods_map = {
    1: "$/mathcal{F}_{gauss}$",
    2: "$/mathcal{F}_{2-rect}$",
    3: "$/mathcal{F}_{3-rect}$",
    23: "$/mathcal{F}_{2,3-rect}$",
}

for dataset in datasets:
    if "BBS" in dataset:
        vqnnf_df = pd.read_csv("BBS_vq_nnf_all_results.csv")

        ddis_df_512 = pd.read_csv("c:/Users/gupta/Desktop/BBS_data/BBS_DDIS_512_results.csv")
        ddis_df_512["n_features"] = 512
        ddis_df_27 = pd.read_csv("c:/Users/gupta/Desktop/BBS_data/BBS_DDIS_results.csv")
        ddis_df_27["n_features"] = 27
        ddis_df = pd.concat([ddis_df_512, ddis_df_27]).reset_index(drop=True)
        ddis_df["dataset"] = "BBS" + ddis_df["exp_cat"].astype(str)
        ddis_df = ddis_df.groupby(["dataset", "n_features"]).mean().reset_index()
        ddis_df["Method"] = "DDIS"

        diwu_df_512 = pd.read_csv("c:/Users/gupta/Desktop/BBS_data/BBS_DIWU_512_results.csv")
        diwu_df_512["n_features"] = 512
        diwu_df_27 = pd.read_csv("c:/Users/gupta/Desktop/BBS_data/BBS_DIWU_results.csv")
        diwu_df_27["n_features"] = 27
        diwu_df = pd.concat([diwu_df_512, diwu_df_27]).reset_index(drop=True)
        diwu_df["dataset"] = "BBS" + diwu_df["exp_cat"].astype(str)
        diwu_df = diwu_df.groupby(["dataset", "n_features"]).mean().reset_index()
        diwu_df["Method"] = "DIWU"

    elif "TLPattr" in dataset:
        vqnnf_df = pd.read_csv("TLPattr_vq_nnf_all_results.csv")

        ddis_df_27 = pd.read_csv("c:/Users/gupta/Desktop/TLPattr/TLPattr_27_DDIS_results.csv")
        ddis_df_27 = ddis_df_27[ddis_df_27["rotation"] == 1]
        ddis_df_27["n_features"] = 27
        ddis_df_27["dataset"] = dataset
        ddis_df_512 = pd.read_csv("c:/Users/gupta/Desktop/TLPattr/TLPattr_512_DDIS_results.csv")
        ddis_df_512 = ddis_df_512[ddis_df_512["rotation"] == 1]
        ddis_df_512["n_features"] = 512
        ddis_df_512["dataset"] = dataset
        ddis_df = pd.concat([ddis_df_512, ddis_df_27]).reset_index(drop=True)
        diwu_df_27 = pd.read_csv("c:/Users/gupta/Desktop/TLPattr/TLPattr_27_DIWU_results.csv")
        diwu_df_27 = diwu_df_27[diwu_df_27["rotation"] == 1]
        diwu_df_27["n_features"] = 27
        diwu_df_27["dataset"] = dataset
        diwu_df_512 = pd.read_csv("c:/Users/gupta/Desktop/TLPattr/TLPattr_512_DIWU_results.csv")
        diwu_df_512 = diwu_df_512[diwu_df_512["rotation"] == 1]
        diwu_df_512["n_features"] = 512
        diwu_df_512["dataset"] = dataset
        diwu_df = pd.concat([diwu_df_512, diwu_df_27]).reset_index(drop=True)

    elif "TinyTLP" in dataset:
        vqnnf_df = pd.read_csv("TinyTLP_vq_nnf_all_results.csv")

        ddis_df_27 = pd.read_csv("c:/Users/gupta/Desktop/TinyTLP_comp/TinyTLP_27_DDIS_results.csv")
        ddis_df_27 = ddis_df_27[ddis_df_27["rotation"] == 0]
        ddis_df_27["n_features"] = 27
        ddis_df_27["dataset"] = dataset
        ddis_df_512 = pd.read_csv("c:/Users/gupta/Desktop/TinyTLP_comp/TinyTLP_512_DDIS_results.csv")
        ddis_df_512 = ddis_df_512[ddis_df_512["rotation"] == 0]
        ddis_df_512["n_features"] = 512
        ddis_df_512["dataset"] = dataset
        ddis_df = pd.concat([ddis_df_512, ddis_df_27]).reset_index(drop=True)

        diwu_df_27 = pd.read_csv("c:/Users/gupta/Desktop/TinyTLP_comp/TinyTLP_27_DIWU_results.csv")
        diwu_df_27 = diwu_df_27[diwu_df_27["rotation"] == 0]
        diwu_df_27["n_features"] = 27
        diwu_df_27["dataset"] = dataset
        diwu_df_512 = pd.read_csv("c:/Users/gupta/Desktop/TinyTLP_comp/TinyTLP_512_DIWU_results.csv")
        diwu_df_512 = diwu_df_512[diwu_df_512["rotation"] == 0]
        diwu_df_512["n_features"] = 512
        diwu_df_512["dataset"] = dataset
        diwu_df = pd.concat([diwu_df_512, diwu_df_27]).reset_index(drop=True)
    else:
        raise ValueError("Invalid dataset")

    vqnnf_df["dataset"] = vqnnf_df["dataset"].str.split("_").str[0]
    vqnnf_df = vqnnf_df.loc[vqnnf_df["dataset"] == dataset]

    vqnnf_df = (
        vqnnf_df.groupby(["dataset", "model", "n_features", "n_codes", "scale", "rect_haar_filters"])
        .mean()
        .reset_index()
    )
    vqnnf_df = vqnnf_df.rename(
        columns={
            "rect_haar_filters": "Filter Sets",
            "n_codes": "Codebook Size",
            "n_features": "Features",
            "scale": "Scales",
            "M_IOU": "MIoU",
            "Success_Rate": "Success Rate",
            "Total_Time": "Total Time (sec.)",
        }
    )
    vqnnf_df.to_csv(f"figures/{dataset}_vqnnf_df.csv", index=False)

    vqnnf_df = vqnnf_df[
        (vqnnf_df["Features"].isin(n_features))
        & (vqnnf_df["Codebook Size"].isin(n_clusters))
        & (vqnnf_df["Filter Sets"].isin(n_haar_features))
        & (vqnnf_df["Scales"].isin(scales))
    ]

    vqnnf_df.loc[vqnnf_df["Scales"] == 1, "Codebook Size"] = vqnnf_df.loc[vqnnf_df["Scales"] == 1, "Codebook Size"] * (
        1 - 0.15
    )
    vqnnf_df.loc[vqnnf_df["Scales"] == 2, "Codebook Size"] = vqnnf_df.loc[vqnnf_df["Scales"] == 2, "Codebook Size"] * (
        1 - 0.0
    )
    vqnnf_df.loc[vqnnf_df["Scales"] == 3, "Codebook Size"] = vqnnf_df.loc[vqnnf_df["Scales"] == 3, "Codebook Size"] * (
        1 + 0.15
    )

    vqnnf_df["Filter Sets"] = vqnnf_df["Filter Sets"].map(methods_map)
    vqnnf_df["Features"] = vqnnf_df["Features"].apply(lambda x: "Color" if x == 27 else "Deep")

    # fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    g = sns.relplot(
        data=vqnnf_df,
        y="MIoU",
        x="Codebook Size",
        hue="Scales",
        style="Filter Sets",
        row="Features",
        palette="deep",
        alpha=0.9,
        s=400,
        height=5,
        aspect=1.5,
        legend=False,
        # edgecolor="face",
        # linewidth=1,
        # edgecolor="face"
        # kind="line",
        # ax=ax
    )
    plt.xscale("log", base=2)

    ## draw a horizontal line at at each plot in the column with values of DDIS and DIWU
    for ii, ax in enumerate(g.axes.flat):
        ax.axhline(
            y=ddis_df.loc[(ddis_df["dataset"] == dataset) & (ddis_df["n_features"] == n_features[ii])]["mious"].values[
                0
            ],
            color=sns.color_palette("deep")[3],
            linestyle="--",
            # label="DDIS",
            linewidth=3,
        )
        ax.axhline(
            y=diwu_df.loc[(ddis_df["dataset"] == dataset) & (diwu_df["n_features"] == n_features[ii])]["mious"].values[
                0
            ],
            color=sns.color_palette("deep")[4],
            linestyle="--",
            # label="DIWU",
            linewidth=3,
        )
        feature = "Color" if ii == 0 else "Deep"
        sns.lineplot(
            data=vqnnf_df[
                (vqnnf_df["Features"] == feature)
                & (vqnnf_df["Scales"] == 1)
                & (vqnnf_df["Filter Sets"] == "$/mathcal{F}_{gauss}$")
            ],
            y="MIoU",
            x="Codebook Size",
            color=sns.color_palette("deep")[0],
            ax=ax,
            linewidth=3,
            linestyle="--",
        )
        sns.lineplot(
            data=vqnnf_df[
                (vqnnf_df["Features"] == feature)
                & (vqnnf_df["Scales"] == 3)
                & (vqnnf_df["Filter Sets"] == "$/mathcal{F}_{2,3-rect}$")
            ],
            y="MIoU",
            x="Codebook Size",
            color=sns.color_palette("deep")[2],
            ax=ax,
            linewidth=3,
            linestyle="--",
        )

    plt.savefig(f"figures/design_choices/{dataset}_mean_miou.png", dpi=500, bbox_inches="tight")
    plt.close()

    g = sns.relplot(
        data=vqnnf_df,
        y="Total Time (sec.)",
        x="Codebook Size",
        hue="Scales",
        style="Filter Sets",
        row="Features",
        # row="Scales",
        palette="deep",
        alpha=0.9,
        s=400,
        height=5,
        aspect=1.5,
        legend=False,
        # edgecolor="face",
        # kind="line",
        # ax=ax
    )
    plt.xscale("log", base=2)

    for ii, ax in enumerate(g.axes.flat):
        feature = "Color" if ii == 0 else "Deep"
        sns.lineplot(
            data=vqnnf_df[
                (vqnnf_df["Features"] == feature)
                & (vqnnf_df["Scales"] == 1)
                & (vqnnf_df["Filter Sets"] == "$/mathcal{F}_{gauss}$")
            ],
            y="Total Time (sec.)",
            x="Codebook Size",
            color=sns.color_palette("deep")[0],
            ax=ax,
            linewidth=3,
            linestyle="--",
        )
        sns.lineplot(
            data=vqnnf_df[
                (vqnnf_df["Features"] == feature)
                & (vqnnf_df["Scales"] == 3)
                & (vqnnf_df["Filter Sets"] == "$/mathcal{F}_{2,3-rect}$")
            ],
            y="Total Time (sec.)",
            x="Codebook Size",
            color=sns.color_palette("deep")[2],
            ax=ax,
            linewidth=3,
            linestyle="--",
        )

    plt.savefig(f"figures/design_choices/{dataset}_mean_time.png", dpi=500, bbox_inches="tight")
    plt.close()
    x = 1
