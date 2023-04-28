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

exp_cats = ["exp4", "exp2", "exp"]


n_features = [64, 128, 256, 512]
n_clusters = [4, 8, 16, 32, 64, 128]
n_haar_features = [1, 3, 5, 6]
scales = [1, 2, 3, 4]
result_csvs = []

for exp_cat in exp_cats:
    result_csv = pd.concat(
        [
            pd.concat(
                [
                    pd.read_csv(
                        f"exps/bbs25/bbs25_iter{i}_dataset_all_time_results_optim_{exp_cat}.csv"
                    ),
                ]
            )
            for i in range(1, 6)
        ]
    )

    result_csv = result_csv[
        (result_csv["n_features"].isin(n_features))
        & (result_csv["n_clusters"].isin(n_clusters))
        & (result_csv["n_haar_features"].isin(n_haar_features))
        & (result_csv["scales"].isin(scales))
    ]
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
    result_csv["Method"] = (
        "Exp-4"
        if exp_cat == "exp"
        else "Exp-1"
        if exp_cat == "exp2"
        else "Cosine"
        if exp_cat == "exp3"
        else "Orig"
    )
    result_csvs.append(result_csv)

    # result_csvs = pd.concat(result_csvs)


result_csvs = pd.concat(result_csvs).reset_index(drop=True)
for feature in n_features:
    df = result_csvs[result_csvs["Features"] == feature]
    sns.relplot(
        data=df,
        x="Clusters",
        y="MIoU",
        hue="Method",
        col="Haar Features",
        row="Scales",
        kind="line",
        alpha=0.8,
    )
    plt.xscale("log", base=2)
plt.show()
x = 1
