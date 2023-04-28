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


sns.set_style("whitegrid")
sns.set_context("notebook")  # paper, notebook, talk, poster


n_features = 512
n_clusters = 64
n_haar_features = 3
scales = 2
n_samples = [1, 2, 3, 4, 5]
class_ids = [1]
random_seeds = [0]

for class_id in class_ids:
    all_results = []
    for n_sample in n_samples:
        result_csv = []
        for random_seed in random_seeds:
            result_csv_seed = pd.read_csv(
                f"exps/malaria/malaria_dataset_resnet18_n_features_{n_features}_n_cluster_{n_clusters}_n_haar_feature_{n_haar_features}_scale_{scales}_n_samples_{n_sample}_random_seed_{random_seed}_class_id_{class_id}_pr_df.csv"
            )
            result_csv_seed["mean_recall"] = pd.cut(result_csv_seed["recall_bins"], 1000).apply(
                lambda x: x.mid
            )
            result_csv_seed = (
                result_csv_seed.groupby("mean_recall")[["recall_bins", "mean_precision"]]
                .mean()
                .dropna()
                .reset_index(drop=True)
            )
            result_csv.append(result_csv_seed)

        result_csv = pd.concat(result_csv)
        result_csv = result_csv.rename(
            columns={"recall_bins": "Recall", "mean_precision": "Precision"}
        )
        result_csv = result_csv[["Recall", "Precision"]]

        # result_csv = result_csv.dropna(how="all").reset_index(drop=True)
        result_csv_mean = (
            result_csv.groupby("Recall").mean().reset_index()
        )  # .sort_values("Recall", ascending=False)
        result_csv_mean = (
            result_csv_mean.dropna().sort_values("Recall", ascending=True).reset_index(drop=True)
        )
        auc = np.trapz(result_csv_mean["Precision"], result_csv_mean["Recall"])
        # auc = -np.trapz(result_csv["Precision"], result_csv["Recall"])
        print(auc)
        result_csv["AUC"] = auc
        result_csv["Samples"] = f"{n_sample}; AUC: {np.round(auc,3):.3f}"

        all_results.append(result_csv)

    all_results = pd.concat(all_results).reset_index(drop=True)

    plt.figure()
    sns.lineplot(
        data=all_results,
        x="Recall",
        y="Precision",
        hue="Samples",
        palette="dark",  # , errorbar="sd",
    )
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(f"exps/malaria/malaria_pr_curve_class_{class_id}.pdf")
plt.show()


x = 1

