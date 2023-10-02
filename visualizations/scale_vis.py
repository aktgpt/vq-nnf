import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")
# sns.set_context("talk")  # paper, notebook, talk, poster
sns.set(font_scale=2)
plt.rcParams["font.family"] = "Arial"

n_features = [27, 512]

dataset_folders = [
    f"c:/Users/gupta/Desktop/TLPattr/TLPattr_comp_scale_{i}_TM_Results"
    for i in ("0.25", "0.33", "0.50", "0.66", "0.75", "1.00")
]

vqnnf_dfs = pd.read_csv("TLPattr_vq_nnf_scale_results2.csv")

methods_map = {
    1: "$\mathcal{F}_{gauss}$",
    2: "$\mathcal{F}_{2-rect}$",
    3: "$\mathcal{F}_{3-rect}$",
    23: "$\mathcal{F}_{2,3-rect}$",
}

for n_feat in n_features:
    all_feat_df = []
    for dataset_folder in dataset_folders:
        dataset_name = dataset_folder.split("/")[-1].replace("_TM_Results", "")
        scale_factor = float(dataset_name.split("_")[-1])

        ddis_df = pd.read_csv(f"{dataset_folder}/DDIS_iter_{n_feat}_results.csv")
        ddis_df["Success Rate"] = ddis_df["ious"] > 0.5
        ddis_df = ddis_df.mean(axis=0).to_frame().transpose()
        ddis_df["Method"] = "DDIS"
        ddis_df["Features"] = n_feat
        ddis_df["Time (s)"] = ddis_df["time"]
        ddis_df["MIoU"] = ddis_df["ious"]
        ddis_df["Scale Factor"] = scale_factor
        all_feat_df.append(ddis_df[["Scale Factor", "Method", "Features", "Time (s)", "MIoU", "Success Rate"]])

        diwu_df = pd.read_csv(f"{dataset_folder}/DIWU_iter_{n_feat}_results.csv")
        diwu_df["Success Rate"] = diwu_df["ious"] > 0.5
        diwu_df = diwu_df.mean(axis=0).to_frame().transpose()
        diwu_df["Method"] = "DIWU"
        diwu_df["Features"] = n_feat
        diwu_df["Time (s)"] = diwu_df["time"]
        diwu_df["MIoU"] = diwu_df["ious"]
        diwu_df["Scale Factor"] = scale_factor
        all_feat_df.append(diwu_df[["Scale Factor", "Method", "Features", "Time (s)", "MIoU", "Success Rate"]])

        scale_to_consider = 3 if n_feat == 512 else 4
        vqnnf_df = vqnnf_dfs.loc[
            (vqnnf_dfs["n_features"] == n_feat)
            & (vqnnf_dfs["dataset"] == dataset_name)
            & (vqnnf_dfs["scale"] == scale_to_consider)
        ]

        vqnnf_df["Method"] = f"$S={scale_to_consider}, $" + vqnnf_df["rect_haar_filters"].map(methods_map)
        vqnnf_df["Features"] = n_feat
        vqnnf_df["Time (s)"] = vqnnf_df["Temp_Match_Time"] + vqnnf_df["Kmeans_Time"]
        vqnnf_df["MIoU"] = vqnnf_df["M_IOU"]
        vqnnf_df["Success Rate"] = vqnnf_df["Success_Rate"]
        vqnnf_df["Scale Factor"] = scale_factor
        all_feat_df.append(vqnnf_df[["Scale Factor", "Method", "Features", "Time (s)", "MIoU", "Success Rate"]])

    all_feat_df = pd.concat(all_feat_df).reset_index(drop=True)

    title_name = "Color Features" if n_feat == 27 else "Deep Features"

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    g = sns.lineplot(
        x="Scale Factor",
        y="MIoU",
        hue="Method",
        data=all_feat_df,
        palette="deep",
        alpha=0.8,
        ax=ax,
        marker="o",
        linewidth=5,
        markersize=10,
    )
    # highlight the best result with a horizontal line
    max_miou = all_feat_df["MIoU"].max()
    g.axhline(max_miou, ls="--", color="k", alpha=0.5, label=f"Best MIoU: {max_miou:.3f}")

    handles, labels = g.get_legend_handles_labels()
    ## increase linewidth of legend
    for h in handles:
        h.set_linewidth(5)

    lgd = g.legend(
        handles=handles,
        labels=labels,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
    )

    # plt.xticks(np.arange(0.2, 1.1, 0.2))
    plt.title(title_name)
    plt.savefig(f"figures/scale/scale_comp_{n_feat}.png", bbox_inches="tight", dpi=500, bbox_extra_artists=(lgd,))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    g = sns.lineplot(
        x="Scale Factor",
        y="Time (s)",
        hue="Method",
        # style="Method",
        data=all_feat_df,
        palette="deep",
        alpha=0.8,
        ax=ax,
        marker="o",
        linewidth=5,
        markersize=10,
        legend=False,
    )
    # plt.yscale("log")

    handles, labels = g.get_legend_handles_labels()
    ## increase linewidth of legend
    # for h in handles:
    #     h.set_linewidth(5)
    # lgd = g.legend(
    #     handles=handles,
    #     labels=labels,
    #     bbox_to_anchor=(1.05, 1),
    #     loc=2,
    #     borderaxespad=0.0,
    # )

    plt.title(title_name)
    plt.savefig(f"figures/scale/scale_time_{n_feat}.png", bbox_inches="tight", dpi=500)  # , bbox_extra_artists=(lgd,))

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    g = sns.lineplot(
        x="Scale Factor",
        y="Time (s)",
        hue="Method",
        data=all_feat_df,
        palette="deep",
        alpha=0.8,
        ax=ax,
        marker="o",
        legend=False,
        linewidth=5,
        markersize=10,
    )
    # plt.yscale("log")
    plt.ylim(0, 2)  # 10 if n_feat == 512 else 2)

    # handles, labels = g.get_legend_handles_labels()
    # ## increase linewidth of legend
    # for h in handles:
    #     h.set_linewidth(5)
    # lgd = g.legend(
    #     handles=handles,
    #     labels=labels,
    #     bbox_to_anchor=(1.05, 1),
    #     loc=2,
    #     borderaxespad=0.0,
    # )
    plt.title(title_name)
    plt.savefig(
        f"figures/scale/scale_time_{n_feat}_zoom.png", bbox_inches="tight", dpi=500
    )  # , bbox_extra_artists=(lgd,))
    plt.close("all")
