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
from scipy import stats

folder = "exps/bbs25/rel_diff"


sns.set_style("whitegrid")
sns.set_context("paper")  # paper, notebook, talk, poster


# all_files = glob.glob(os.path.join(folder, "*.csv"))
# all_files = [f for f in all_files if "all_time" not in f]

n_features = [512]
n_clusters = [128]
n_haar_features = [1, 3, 5, 6]
scales = [1, 2, 3, 4]

cols = ["BC", "DEF", "FM", "IPR", "IV", "LR", "MB", "OCC", "OPR", "OV", "SV"]


# get params from file name
result_csvs = []
all_methods = []
for scale in scales:
    for n_haar_feature in n_haar_features:
        result_csv = pd.concat(
            [
                pd.read_csv(file_name)
                for file_name in sorted(
                    glob.glob(
                        os.path.join(
                            folder,
                            f"bbs25_iter*_dataset_resnet18_n_features_27_n_cluster_128_n_haar_feature_{n_haar_feature}_scale_{scale}_iou_sr.csv",
                        )
                    )
                )
            ]
        ).reset_index(drop=True)
        # result_csv["n_haar_features"] = n_haar_feature
        # result_csv["scales"] = scale
        # result_csv["Category"] = f"Scale:{scale}, Haar Features:{n_haar_feature}"
        result_csv.rename(
            columns={"ious": f"Scale: {scale}, Haar Features: {n_haar_feature}"}, inplace=True,
        )
        all_methods.append(f"Scale: {scale}, Haar Features: {n_haar_feature}")

        if len(result_csvs) == 0:
            result_csv = result_csv[
                [
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
                    "img_size",
                    "temp_size",
                    f"Scale: {scale}, Haar Features: {n_haar_feature}",
                ]
            ]
            result_csvs = result_csv

        else:
            result_csvs[f"Scale: {scale}, Haar Features: {n_haar_feature}"] = result_csv[
                f"Scale: {scale}, Haar Features: {n_haar_feature}"
            ]

# get best category
mean_df = result_csvs[all_methods].mean(axis=0)
best_cat = mean_df.idxmax()
result_csvs[f"Best, {best_cat}"] = result_csvs[best_cat]

ddis_df = pd.concat(
    [
        pd.read_csv(file_name)
        for file_name in sorted(
            glob.glob(os.path.join("exps/comparison/bbs25/DDIS_iter*_results.csv"))
        )
    ]
).reset_index(drop=True)
result_csvs["DDIS"] = ddis_df["ious"]

diwu_df = pd.concat(
    [
        pd.read_csv(file_name)
        for file_name in sorted(
            glob.glob(os.path.join("exps/comparison/bbs25/DIWU_iter*_results.csv"))
        )
    ]
).reset_index(drop=True)
result_csvs["DIWU"] = diwu_df["ious"]

plot_df = []
for col in cols:
    col_df = result_csvs.iloc[np.where(result_csvs[col] == 1)].reset_index(drop=True)
    ## get best method by mann-whitney u test
    best_col_cat = col_df[all_methods].mean(axis=0).idxmax()
    col_rel_df = col_df[["DDIS", "DIWU", best_cat, best_col_cat]].melt(
        var_name="Method", value_name="IOU"
    )
    col_rel_df["Attributes"] = col
    plot_df.append(col_rel_df)

    # result, pval = stats.mannwhitneyu(col_df[best_col_cat], col_df["DIWU"], alternative="greater")
    # print(f"{col} vs DDIS: {pval}")
    # result, pval = stats.mannwhitneyu(col_df[best_col_cat], col_csv["DDIS"], alternative="greater")
    # print(f"{col} vs DIWU: {pval}")

plot_df = pd.concat(plot_df).reset_index(drop=True)
hue_order = sorted(plot_df["Method"].unique())
sns.barplot(data=plot_df, x="Attributes", y="IOU", hue="Method", hue_order=hue_order)
plt.show()

# mean_df = result_csv.groupby(["Category"]).mean().reset_index()
# best_cat = mean_df.iloc[np.argmax(mean_df["ious"])]["Category"]


# ddis_df["Category"] = "DDIS"
# ddis_df[["BC", "DEF", "FM", "IPR", "IV", "LR", "MB", "OCC", "OPR", "OV", "SV"]] = result_csv[
#     ["BC", "DEF", "FM", "IPR", "IV", "LR", "MB", "OCC", "OPR", "OV", "SV"]
# ][: len(ddis_df)]
# result_csv = pd.concat([result_csv, ddis_df]).reset_index(drop=True)

# diwu_df = pd.concat(
#     [
#         pd.read_csv(file_name)
#         for file_name in glob.glob(os.path.join("exps/comparison/bbs25/DIWU_iter*_results.csv"))
#     ]
# ).reset_index(drop=True)
# diwu_df["Category"] = "DIWU"
# diwu_df[["BC", "DEF", "FM", "IPR", "IV", "LR", "MB", "OCC", "OPR", "OV", "SV"]] = result_csv[
#     ["BC", "DEF", "FM", "IPR", "IV", "LR", "MB", "OCC", "OPR", "OV", "SV"]
# ][: len(diwu_df)]
# result_csv = pd.concat([result_csv, diwu_df]).reset_index(drop=True)


# plot_df = pd.concat(
#     [result_csv.iloc[np.where(result_csv[col] == 1)].reset_index(drop=True) for col in cols]
# ).reset_index(drop=True)
plot_df = []
mean_col_df = []
col_mean_std_df = []
for col in cols:
    cat_df = result_csv.iloc[np.where(result_csv[col] == 1)].reset_index(drop=True)
    cat_df["Occlusion"] = col
    mean = cat_df["ious"].groupby(cat_df["Category"]).mean()
    std = cat_df["ious"].groupby(cat_df["Category"]).std()
    # cats = cat_df["Category"].unique()
    merge_df = pd.concat([mean, std], axis=1).reset_index()
    merge_df.columns = ["Category", "Mean", "Std"]
    merge_df["Occlusion"] = col
    # get best column other than DDIS and DIWU
    best_col_cat = merge_df.iloc[np.argmax(merge_df["Mean"].iloc[2:]) + 2]["Category"]

    cat_keep = ["DDIS", "DIWU", best_cat, best_col_cat]
    cat_df = cat_df[cat_df["Category"].isin(cat_keep)].reset_index(drop=True)
    # col_mean_std_df.append(
    #     {"Category": cat_df["Category"][0], "Occlusion": col, "Mean": mean, "Std": std},
    #     # ignore_index=True,
    # )
    mean_col_df.append(merge_df[merge_df["Category"].isin(cat_keep)].reset_index(drop=True))
    plot_df.append(cat_df)
plot_df = pd.concat(plot_df).reset_index(drop=True)
mean_col_df = pd.concat(mean_col_df).reset_index(drop=True)
col_mean_std_df = pd.DataFrame(col_mean_std_df)
sns.boxplot(
    x="Occlusion", y="ious", hue="Category", data=plot_df, palette=sns.color_palette(cc.glasbey, 10)
)  # , cut=0)

for col in cols:
    # plt.figure()
    df = result_csv.iloc[np.where(result_csv[col] == 1)].reset_index(drop=True)
    sns.violinplot(x=col, y="ious", hue="Category", data=df, cut=0)
    # plt.legend(loc="upper left", bbox_to_anchor=(1.1, 1.1))
plt.show()
x = 1
