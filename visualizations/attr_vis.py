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
from matplotlib.patches import Patch

sns.set_style("whitegrid")
# sns.set_context("notebook")  # paper, notebook, talk, poster
sns.set(font_scale=1.5)
plt.rcParams["font.family"] = "Arial"

occlusion_map = {
    "bc": "Background Clutter",
    "fm": "Fast Motion",
    "iv": "Illumination Variation",
    "oc": "Partial Occlusion",
    "ov": "Out-of-View",
    "sv": "Scale Variation",
}
n_features = [27, 512]

dataset_df = pd.read_csv("dataset_annotations/tlpattr_dataset.csv")
print(np.unique(dataset_df["occlusions"].values, return_counts=True))

result_folder = "c:/Users/gupta/Desktop/TLPattr/TLPattr_comp_scale_1.00_TM_Results"

all_df = []
for n_feat in n_features:
    ddis_df = pd.read_csv(f"{result_folder}/DDIS_iter_{n_feat}_results.csv")
    ddis_df["Features"] = "Color" if n_feat == 27 else "Deep"
    ddis_df["Method"] = "DDIS"
    ddis_df["Occlusion"] = dataset_df["occlusions"].map(occlusion_map)
    all_df.append(ddis_df)

    diwu_df = pd.read_csv(f"{result_folder}/DIWU_iter_{n_feat}_results.csv")
    diwu_df["Features"] = "Color" if n_feat == 27 else "Deep"
    diwu_df["Method"] = "DIWU"
    diwu_df["Occlusion"] = dataset_df["occlusions"].map(occlusion_map)
    all_df.append(diwu_df)

    vqnnf_df = pd.read_csv(f"TLPattr_vq_nnf_all_results.csv")
    vqnnf_df = vqnnf_df.loc[vqnnf_df["n_features"] == n_feat]
    our_best = vqnnf_df.loc[(vqnnf_df["M_IOU"] == vqnnf_df["M_IOU"].max())]
    best_vqnnf_df = pd.read_csv(
        f"{result_folder}/VQ_NNF/model_{our_best['model'].values[0]}_n_feats_{our_best['n_features'].values[0].astype(int)}"
        + f"_n_codes_{our_best['n_codes'].values[0].astype(int)}"
        + f"_haar_filts_{our_best['rect_haar_filters'].values[0].astype(int)}"
        + f"_scale_{our_best['scale'].values[0].astype(int)}/iou_sr.csv"
    )
    best_vqnnf_df["Features"] = "Color" if n_feat == 27 else "Deep"
    best_vqnnf_df["Method"] = "Ours"
    best_vqnnf_df["Occlusion"] = dataset_df["occlusions"].map(occlusion_map)
    all_df.append(best_vqnnf_df)

all_df = pd.concat(all_df).reset_index(drop=True)
all_df["MIoU"] = all_df["ious"]


occlusions = np.unique(all_df["Occlusion"].values)
for occlusion in occlusions:
    occ_df = all_df.loc[all_df["Occlusion"] == occlusion].reset_index(drop=True)

    g = sns.barplot(
        data=occ_df,
        x="Features",
        y="MIoU",
        hue="Method",
        hue_order=["DDIS", "DIWU", "Ours"],
        palette="deep",
    )
    g.set_title(f"Attribute: {occlusion}")
    g.set_ylim(0, 1)

    # handles, labels = g.get_legend_handles_labels()
    # g.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

    plt.savefig(f"figures/attributes/{occlusion}_miou.png", bbox_inches="tight", dpi=300)
    plt.close()
    # g = sns.barplot(
    #     data=occ_df,


x = 1
