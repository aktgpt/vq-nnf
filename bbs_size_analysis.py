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


datasets_filepath = [f"dataset_annotations/bbs25_iter{i}_dataset.csv" for i in range(1, 6)]

exp_folder = "exps/bbs25"
n_features = [256]
n_clusters = [16, 32]
n_haar_features = [1, 3, 4, 6]
scales = [1, 2, 3, 4]

columns = [
    "n_clusters",
    "n_haar_features",
    "scales",
    "template_w",
    "template_h",
    "BC",
    "DEF",
    "FM",
    "IPR",
    "IV",
    "LR",
    "MB",
    "OCC",
    "OPR",
    "OV",
    "SV",
]


for i, dataset_filepath in enumerate(datasets_filepath):
    # dataset_csv = pd.read_csv(dataset_filepath)
    # all_df[columns] = dataset_csv[columns]
    for n_feature in n_features:
        for n_cluster in n_clusters:
            for scale in scales:
                all_df = pd.DataFrame(columns=columns)

                for n_haar_feature in n_haar_features:
                    df = pd.DataFrame(columns=["n_clusters", "n_haar_features", "scales"])
                    dataset_csv = pd.read_csv(dataset_filepath)
                    df[columns[3:]] = dataset_csv[columns[3:]]
                    dataset_name = dataset_filepath.split("/")[-1].split(".")[0]

                    exp_name = f"{dataset_name}_resnet18_n_features_{n_feature}_n_cluster_{n_cluster}_n_haar_feature_{n_haar_feature}_scale_{scale}"
                    try:
                        df_iou = pd.read_csv(f"{exp_folder}/{exp_name}_iou_sr.csv")
                        df["iou"] = df_iou["iou"]
                        df["n_clusters"] = n_cluster
                        df["n_haar_features"] = n_haar_feature
                        df["scales"] = scale
                        all_df = all_df.append(df)
                    except:
                        print(f"exp {exp_name} not found")

                all_df = all_df.dropna()
                all_df = all_df.reset_index(drop=True)
                all_df["template_area"] = (
                    np.round(
                        np.sqrt(
                            (all_df["template_w"] * all_df["template_h"]).values.astype("float")
                            / 100
                        )
                    )
                    * 10
                )

                plt.figure()
                plt.title(
                    f"n_cluster: {n_cluster} n_haar_features: {n_haar_feature} scale: {scale}"
                )
                g = sns.boxplot(
                    x="template_area",
                    y="iou",
                    hue="n_haar_features",
                    # style="n_haar_features",
                    data=all_df,
                    palette="dark",
                    # alpha=0.7,
                )
            # plt.figure()
            # g = sns.boxplot(
            #     x="template_area",
            #     y="iou",
            #     hue="scales",
            #     # style="n_haar_features",
            #     data=all_df,
            #     palette="dark",
            # )
            # g.set(xscale="log")

        plt.show()
x = 1

