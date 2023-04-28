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

plt.rcParams["font.family"] = "Arial"
sns.set_style("whitegrid")
sns.set_context("talk")  # paper, notebook, talk, poster
folder = "exps_final"

dataset_df = pd.read_csv("dataset_annotations/tlpattr_dataset.csv")
print(np.unique(dataset_df["occlusions"].values, return_counts=True))

methods = ["DDIS", "DIWU"]
n_features = [27, 512]
comp_df = []
for method in methods:
    for n_feature in n_features:
        df = pd.read_csv(
            f"exps_final/comp/tlpattr/TLPattr_comp_scale_1.00_TM_Results/{method}_iter_{n_feature}_results.csv"
        )
        df["Method"] = method
        df["Features"] = n_feature
        df["Occulusion"] = dataset_df["occlusions"].values
        comp_df.append(df)
comp_df = pd.concat(comp_df).reset_index(drop=True)
comp_df["img_size"] = comp_df["img_size"].fillna(comp_df["img_size"][0])

n_clusters = 128


# folder = "exps/tlpattr/gauss"

# get params from file name
result_csvs = []
all_methods = []
all_methods.append(methods)
for n_feature in n_features:
    n_haar_features = 2
    scales = 3 if n_feature == 27 else 2

    result_csv = pd.read_csv(
        f"exps_final/tlpattr/attr/tlpattr_dataset_resnet18_n_features_{n_feature}_n_cluster_{n_clusters}_n_haar_feature_{n_haar_features}_scale_{scales}_iou_sr.csv"
    )
    result_csv["time"] = result_csv["time"] + result_csv["kmeans_time"]
    result_csv["Features"] = n_feature
    result_csv["Occulusion"] = dataset_df["occlusions"].values

    result_csv["Method"] = "Ours"  # f"$scales={scales}, haar={n_haar_features}$"
    result_csvs.append(
        result_csv[["Method", "Features", "Occulusion", "ious", "time", "temp_size", "img_size"]]
    )
    all_methods.append(f"$scales={scales}, haar={n_haar_features}$")

result_csvs = pd.concat(result_csvs).reset_index(drop=True)

all_df = pd.concat([comp_df, result_csvs]).reset_index(drop=True)
all_df["Occulusion"] = all_df["Occulusion"].apply(
    lambda x: "Background Clutter"
    if x == "bc"
    else "Fast Motion"
    if x == "fm"
    else "Illumination Variation"
    if x == "iv"
    else "Partial Occlusion"
    if x == "oc"
    else "Out-of-View"
    if x == "ov"
    else "Scale Variation"
    if x == "sv"
    else "Unknown"
)
all_df = all_df.rename(
    columns={
        "occlusion": "Occulusion",
        "ious": "MIoU",
        "time": "Time",
        "temp_size": "Template Size",
        "img_size": "Image Size",
        "Features": "Feature Dimensions",
    }
)

all_df["Success Rate"] = all_df["MIoU"] > 0.5

unique_occlusions = all_df["Occulusion"].unique()
for occlusion in unique_occlusions:
    occ_df = all_df[all_df["Occulusion"] == occlusion].reset_index(drop=True)
    print(f"Attribute: {occlusion}, {len(occ_df)}")
    g = sns.barplot(
        x="Feature Dimensions",
        y="MIoU",
        hue="Method",
        data=occ_df,
        palette="colorblind",
        # errorbar="sd",
    )
    # g = sns.barplot(
    #     x="Feature Dimensions",
    #     y="Success Rate",
    #     hue="Method",
    #     data=occ_df,
    #     palette="colorblind",
    #     alpha=0.5,
    #     # errorbar="sd",
    # )
    g.set_title(f"Attribute: {occlusion}")
    g.set_ylim(0, 1)
    plt.savefig(f"exps_final/figures/attr/attr_vis_{occlusion}.pdf", dpi=500, bbox_inches="tight")
    plt.close()

all_df["$L\cdot l$"] = np.round(all_df["Template Size"] * all_df["Image Size"] / 1e6)  # * 1e2
all_df["ious1"] = np.round(all_df["MIoU"], 1)
all_df["temp_size1"] = np.round(all_df["Template Size"] / 1e2)  # * 1e2

g = sns.lineplot(
    x="temp_size1", y="ious1", hue="Method", data=all_df, palette="colorblind"  # , errorbar=None
)
g.set(xscale="log", yscale="log")

plt.xscale("log", base=10)


g = sns.lineplot(
    x="$L\cdot l$", y="ious1", hue="Method", data=all_df, palette="colorblind", errorbar=None
)
plt.xscale("log", base=10)
# plt.yscale("log", base=10)

i = 0
for scale in scales:
    for n_haar_feature in n_haar_features:
        result_csv = pd.read_csv(
            f"exps_final/tlpattr/tlpattr_dataset_resnet18_n_features_{n_features}_n_cluster_{n_clusters}_n_haar_feature_{n_haar_feature}_scale_{scale}_iou_sr.csv"
        )

        result_csv["Method"] = f"s={scale},h={n_haar_feature},k={kernel_size}"
        result_csvs.append(result_csv[["Method", "occlusion", "ious", "temp_size"]])

        if i == 0:
            method_df = pd.DataFrame(
                {
                    "occlusion": result_csv["occlusion"].values,
                    "temp_size": result_csv["temp_size"].values,
                    f"s={scale},h={n_haar_feature},k={kernel_size}": result_csv["ious"].values,
                }
            )
        else:
            try:
                method_df[f"s={scale},h={n_haar_feature},k={kernel_size}"] = result_csv[
                    "ious"
                ].values
            except:
                continue

        # result_csv.rename(
        #     columns={"ious": f"Scale: {scale}, Haar Features: {n_haar_feature}"}, inplace=True,
        # )
        all_methods.append(f"s={scale},h={n_haar_feature},k={kernel_size}")
        i += 1

result_csvs = pd.concat(result_csvs).reset_index(drop=True)

# g = sns.barplot(
#     x="occlusion",
#     y="sr",
#     hue="Method",
#     # k_depth="trustworthy",
#     # scale="area",
#     data=result_csvs,
#     palette=sns.color_palette(cc.glasbey, len(all_methods)),
# )
# sns.move_legend(g, loc="upper left")
ddis_df = pd.read_csv(
    f"exps/tlpattr/gauss/DDIS_iter_results.csv"
    if n_features == 27
    else f"exps/tlpattr/gauss/DDIS_iter_deep_results.csv"
)
ddis_df["Method"] = "DDIS"
ddis_df["temp_size"] = result_csvs["temp_size"][: len(ddis_df)]
ddis_df["occlusion"] = result_csvs["occlusion"][: len(ddis_df)]
method_df["DDIS"] = ddis_df["ious"].values
all_methods.append("DDIS")

diwu_df = pd.read_csv(
    "exps/tlpattr/gauss/DIWU_iter_results.csv"
    if n_features == 27
    else "exps/tlpattr/gauss/DIWU_iter_deep_results.csv"
)
diwu_df["Method"] = "DIWU"
diwu_df["temp_size"] = result_csvs["temp_size"][: len(diwu_df)]
diwu_df["occlusion"] = result_csvs["occlusion"][: len(diwu_df)]
method_df["DIWU"] = diwu_df["ious"].values
all_methods.append("DIWU")

result_csvs = pd.concat(
    [
        result_csvs,
        ddis_df[["Method", "ious", "occlusion", "temp_size"]],
        diwu_df[["Method", "ious", "occlusion", "temp_size"]],
    ]
).reset_index(drop=True)


result_csvs["sr"] = result_csvs["ious"] > 0.5

result_csvs["temp_size"] = np.log10(result_csvs["temp_size"])
interval = pd.cut(result_csvs["temp_size"], bins=10)
result_csvs["temp_size"] = interval.apply(lambda x: x.mid).values.astype(np.float32)
# g = sns.barplot(
#     data=result_csvs,
#     y="sr",
#     x="temp_size",
#     hue="Method",
#     # multiple="stack",
#     # kind="kde",
#     palette=sns.color_palette(cc.glasbey, len(all_methods)),
# )


## get iou correlation between methods for each occlusion category
for i, occlusion in enumerate(np.unique(result_csvs["occlusion"])):
    cat_df = result_csvs[result_csvs["occlusion"] == occlusion]
    cat_df = cat_df.drop(["occlusion"], axis=1)

    cat_df["temp_size"] = np.log10(cat_df["temp_size"])
    interval = pd.cut(cat_df["temp_size"], bins=10)
    cat_df["temp_size"] = interval.apply(lambda x: x.mid).values.astype(np.float32)
    # plt.figure()
    # g = sns.barplot(
    #     data=cat_df,
    #     y="sr",
    #     x="temp_size",
    #     hue="Method",
    #     # multiple="stack",
    #     # kind="kde",
    #     palette=sns.color_palette(cc.glasbey, len(all_methods)),
    #     # bins=100,
    # )
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    # plt.title(f"Occlusion: {occlusion}")

    # cat_df = pd.pivot_table(cat_df, values="sr", index=["Method"], aggfunc=np.mean)
    ## mann whitney u test for each method pair
    method_cat_df = method_df[method_df["occlusion"] == occlusion]
    method_cat_df = method_cat_df.drop(["occlusion"], axis=1)

    # get correlation of methods with temp_size
    corr_df = method_cat_df.corr()

    pval_mat = np.zeros((len(all_methods), len(all_methods)))
    for i, method1 in enumerate(all_methods):
        for j, method2 in enumerate(all_methods):
            _, pval_mat[i, j] = stats.mannwhitneyu(
                method_cat_df[method1].values,
                method_cat_df[method2].values,
                alternative="greater",
                use_continuity=False,
            )
    pval_mat = pd.DataFrame(pval_mat, columns=all_methods, index=all_methods)
    # ## draw heatmap
    # plt.figure(figsize=(10, 10))
    # sns.heatmap(pval_mat, annot=True, fmt=".2f", cmap="Blues")

plt.show()
# result_csvs["Scale"] = (
#     result_csvs["Method"].apply(lambda x: int(x.split(",")[0].split(":")[1])).astype(str)
# )
# result_csvs["Haar Features"] = (
#     result_csvs["Method"].apply(lambda x: int(x.split(",")[1].split(":")[1])).astype(str)
# )
# result_csvs["Kernel Size"] = (
#     result_csvs["Method"].apply(lambda x: int(x.split(",")[2].split(":")[1])).astype(str)
# )

# make barplot for each occlusion category separately with different hue for each method
# draw a horizontal line for the baseline DIWU and DDIS
result_csvs["occlusion"] = result_csvs["occlusion"].apply(
    lambda x: "Background Clutter"
    if x == "bc"
    else "Fast Motion"
    if x == "fm"
    else "Illumination Variation"
    if x == "iv"
    else "Partial Occlusion"
    if x == "oc"
    else "Out-of-View"
    if x == "ov"
    else "Scale Variation"
    if x == "sv"
    else "Unknown"
)


for i, occlusion in enumerate(np.unique(result_csvs["occlusion"])):
    cat_df = result_csvs[result_csvs["occlusion"] == occlusion]
    plt.figure(figsize=(16, 10), tight_layout=True)
    g = sns.barplot(
        x="Method",
        y="sr",
        data=cat_df,
        palette=sns.color_palette(cc.glasbey, len(all_methods)),
        alpha=0.5,
        label="SR",
    )
    g = sns.barplot(
        x="Method",
        y="ious",
        data=cat_df,
        palette=sns.color_palette(cc.glasbey, len(all_methods)),
        alpha=0.8,
        label="IoU",
    )
    g.set_title(occlusion)
    g.set_ylim(0, 1)
    g.axhline(
        y=cat_df[cat_df["Method"] == "DDIS"]["sr"].mean(),
        color=sns.color_palette(cc.glasbey, len(all_methods))[-2],
        linestyle="-",
        alpha=0.8,
        # linewidth=2,
        label="DDIS SR",
    )
    g.axhline(
        y=cat_df[cat_df["Method"] == "DDIS"]["ious"].mean(),
        color=sns.color_palette(cc.glasbey, len(all_methods))[-2],
        linestyle="--",
        alpha=0.8,
        # linewidth=2,
        label="DDIS IoU",
    )
    g.axhline(
        y=cat_df[cat_df["Method"] == "DIWU"]["sr"].mean(),
        color=sns.color_palette(cc.glasbey, len(all_methods))[-1],
        linestyle="-",
        alpha=0.8,
        # linewidth=2,
        label="DIWU SR",
    )
    g.axhline(
        y=cat_df[cat_df["Method"] == "DIWU"]["ious"].mean(),
        color=sns.color_palette(cc.glasbey, len(all_methods))[-1],
        linestyle="--",
        alpha=0.8,
        # linewidth=2,
        label="DIWU IoU",
    )
    plt.xticks(rotation=90)
    plt.legend()
    plt.savefig(os.path.join(folder, "plots", f"{occlusion}_{n_features}.png"))
    plt.close()

# plt.show()

for i, occlusion in enumerate(np.unique(result_csvs["occlusion"])):
    cat_df = result_csvs[result_csvs["occlusion"] == occlusion]
    cat_df["Scale"] = cat_df["Scale"].astype(str)
    g = sns.barplot(x="Method", y="sr", data=cat_df,)

    fig, ax = plt.subplots(1, 2)
    g = sns.barplot(
        x="Scale",
        y="sr",
        hue="Method",
        data=cat_df,
        palette=sns.color_palette(cc.glasbey, len(all_methods)),
        ax=ax[0],
        # dodge=False,
    )
    g.set_title(occlusion)
    g.set_ylim(0, 1)
    g.legend_.remove()
    # change_width(g, 0.25)
    g = sns.barplot(
        x="Haar Features",
        y="ious",
        hue="Method",
        data=cat_df,
        palette=sns.color_palette(cc.glasbey, len(all_methods)),
        ax=ax[1],
        # dodge=False,
    )
    g.set_title(occlusion)
    g.set_ylim(0, 1)
    g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    # g.legend_.remove()
    # change_width(g, 0.25)

plt.show()


# make barplot for each occlusion category separately
fig, ax = plt.subplots(len(np.unique(result_csvs["occlusion"])), 1, sharey=True)
for i, occlusion in enumerate(np.unique(result_csvs["occlusion"])):
    cat_df = result_csvs[result_csvs["occlusion"] == occlusion]
    cat_df["Scale"] = cat_df["Scale"].astype(str)
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    g = sns.barplot(
        x="Scale",
        y="sr",
        hue="Method",
        data=cat_df,
        ax=ax[i],  # ax[i // 3, i % 3],
        palette=sns.color_palette(cc.glasbey, len(all_methods)),
        # orient="h",
    )
    ax[i].set_title(occlusion)
    ax[i].set_ylim(0, 1)
    ax[i].legend_.remove()
    # ax[i // 3, i % 3].set_title(occlusion)
    # ax[i // 3, i % 3].set_ylim(0, 1)
    # if i != 0:
    #     ax[i // 3, i % 3].legend_.remove()
# plt.show()

fig, ax = plt.subplots(len(np.unique(result_csvs["occlusion"])) // 3, 3, figsize=(20, 10))
for i, occlusion in enumerate(np.unique(result_csvs["occlusion"])):
    cat_df = result_csvs[result_csvs["occlusion"] == occlusion]
    g = sns.barplot(
        x="Haar Features",
        y="sr",
        hue="Method",
        data=cat_df,
        ax=ax[i // 3, i % 3],
        palette=sns.color_palette(cc.glasbey, len(all_methods)),
        # orient="h",
    )
    ax[i // 3, i % 3].set_title(occlusion)
    ax[i // 3, i % 3].set_ylim(0, 1)
    if i != 0:
        ax[i // 3, i % 3].legend_.remove()
plt.show()


mean_cat_df = result_csvs.groupby(["occlusion", "Method"]).mean().reset_index()
mean_cat_df["Scale"] = mean_cat_df["Method"].apply(lambda x: int(x.split(",")[0].split(":")[1]))
mean_cat_df["Haar Features"] = mean_cat_df["Method"].apply(
    lambda x: int(x.split(",")[1].split(":")[1])
)
mean_cat_df["Kernel Size"] = mean_cat_df["Method"].apply(
    lambda x: int(x.split(",")[2].split(":")[1])
)
fig, ax = plt.subplots(len(np.unique(mean_cat_df["occlusion"])) // 3, 3, figsize=(20, 10))
for i, occlusion in enumerate(np.unique(mean_cat_df["occlusion"])):
    cat_df = mean_cat_df[mean_cat_df["occlusion"] == occlusion]
    sns.barplot(
        x="Method",
        y="sr",
        data=cat_df,
        ax=ax[i // 3, i % 3],
        palette=sns.color_palette(cc.glasbey, len(all_methods)),
    )

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
sns.boxplot(data=plot_df, x="Attributes", y="IOU", hue="Method", hue_order=hue_order)
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
