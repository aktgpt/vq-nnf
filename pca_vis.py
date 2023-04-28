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


plt.rcParams["font.family"] = "Arial"
sns.set_style("whitegrid")
sns.set_context("talk")  # paper, notebook, talk, poster
folder = "exps_final"


n_features = [27, 512]

result_csvs = []
n_clusters = [128]
n_haar_features = [1, 2, 3, 23]


datasets = ["bbs25", "bbs50", "bbs100", "tiny_tlp", "tlpattr"]  # ]

all_df = []
for dataset in datasets:
    result_csv = (
        pd.concat(
            [
                pd.read_csv(x)
                for x in glob.glob(
                    f"exps_final/{dataset}/pca/{dataset}_iter*_dataset_all_time_results_k3s2haar_pca.csv"
                )
            ],
        ).reset_index(drop=True)
        if "bbs" in dataset
        else pd.read_csv(
            f"exps_final/{dataset}/pca/{dataset}_dataset_all_time_results_k3s2haar_pca.csv"
        )
    )
    result_csv.fillna(512, inplace=True)
    result_csv = result_csv.groupby(["pca_dim"]).mean().reset_index()
    result_csv["Time (s)"] = result_csv["Temp_Match_Time"] + result_csv["Kmeans_Time"]

    result_csv["Dataset"] = dataset
    all_df.append(result_csv)

all_dfs = pd.concat(all_df).reset_index(drop=True)[["Dataset", "pca_dim", "M_IOU", "Time (s)"]]
all_dfs1 = all_dfs.pivot(
    index="pca_dim", columns="Dataset", values=["M_IOU", "Time (s)"]
).reset_index()
all_dfs1 = all_dfs1.sort_values(by="pca_dim", ascending=False)
all_dfs1.to_csv("exps_final/stats/pca.csv", index=False)
