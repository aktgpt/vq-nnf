import argparse
import os
import random
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTHONHASHSEED"] = str(69)

import torch
import torchvision.ops.boxes as bops
from skimage.feature import peak_local_max
from sklearn import metrics
from feature_extraction import PixelFeatureExtractor
from matchers.template_matching import HaarTemplateMatcher

matplotlib.use("Agg")
sns.set_style("whitegrid")
sns.set_context("notebook")  # paper, notebook, talk, poster


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


image_size = (512, 512)
patch_size = (72, 72)
min_val, max_val = np.array([7628]), np.array([10827])


def get_min_max_percentile(df, percentile=0.5):
    images = []
    for i, row in df.iterrows():
        image = cv2.imread(
            os.path.join(
                "/mnt/hdd1/users/aktgpt/datasets/template_matching/cilia/images", row["image_id"],
            ),
            -1,
        )
        images.append(image[np.newaxis, :])

    images = np.concatenate(images, axis=0)
    min_val = np.percentile(images, percentile)
    max_val = np.percentile(images, 100 - percentile)
    return min_val, max_val


def normalize_image(image, min_val, max_val):
    image = (image - min_val) / (max_val - min_val)
    image = np.clip(image, 0, 1)
    return image


def main(
    exp_folder,
    dataset_filepath,
    model_name,
    n_features,
    n_clusters,
    n_haar_features,
    scale,
    n_samples,
):
    dataset_csv = pd.read_csv(dataset_filepath, index_col=0)
    unique_images = np.unique(dataset_csv["image_id"].values)

    dataset_csv["freq"] = len(dataset_csv) / dataset_csv.groupby("image_id")["image_id"].transform(
        "count"
    )
    sampled_df = dataset_csv.sample(n=50 * n_samples, weights="freq").reset_index(drop=True)
    sampled_df.to_csv(f"{exp_folder}_samples.csv")

    feature_extractor = PixelFeatureExtractor(model_name=model_name, num_features=n_features)

    tqdm_dataset = tqdm(desc="Images Processed", total=len(sampled_df), position=0)
    auc_desc = tqdm(total=0, position=1, bar_format="{desc}")
    template_box_desc = tqdm(total=0, position=2, bar_format="{desc}")

    pr_df = pd.DataFrame({"recall_bins": np.round(np.arange(0.0005, 1.0, 0.001), 4)})
    for i in range(0, len(sampled_df), n_samples):
        rows = sampled_df.iloc[i : i + n_samples]
        # images_ids = np.unique(rows["image_id"].values)
        # template_bboxes = rows[["x1", "y1", "x2", "y2"]].values

        # temp_w, temp_h = template_bboxes[0, 2:] - template_bboxes[0, :2]

        template_matchers = []
        query_images = unique_images.copy()
        for _, row in rows.iterrows():
            image_id = row["image_id"]
            image = cv2.cvtColor(
                cv2.resize(
                    cv2.imread(
                        os.path.join(
                            "/mnt/hdd1/users/aktgpt/datasets/template_matching/cilia/images",
                            image_id,
                        ),
                        -1,
                    ),
                    image_size,
                ),
                cv2.COLOR_GRAY2RGB,
            )
            image = (normalize_image(image, min_val, max_val) * 255).astype(np.uint8)

            image_features = feature_extractor.get_features(image)

            template_bbox = row[["x1", "y1", "x2", "y2"]].values
            template_bbox[template_bbox < 0] = 0
            template_features = image_features[
                :, template_bbox[1] : template_bbox[3], template_bbox[0] : template_bbox[2]
            ]
            template_box_desc.set_description(
                f"Template Bbox: {template_bbox}, Template Features: {template_features.shape}"
            )

            template_matcher = HaarTemplateMatcher(
                template_features,
                patch_size,
                n_clusters=n_clusters,
                n_haar_features=n_haar_features,
                scales=scale,
            )
            template_matchers.append(template_matcher)
            query_images = query_images[query_images != image_id]

        query_images_df = dataset_csv[dataset_csv["image_id"].isin(query_images)]

        all_predictions_df = []
        for query_image_id in query_images:
            query_image_df = dataset_csv[dataset_csv["image_id"] == query_image_id]
            query_image = cv2.cvtColor(
                cv2.resize(
                    cv2.imread(
                        os.path.join(
                            "/mnt/hdd1/users/aktgpt/datasets/template_matching/cilia/images",
                            query_image_id,
                        ),
                        -1,
                    ),
                    image_size,
                ),
                cv2.COLOR_GRAY2RGB,
            )
            query_image = (normalize_image(query_image, min_val, max_val) * 255).astype(np.uint8)

            query_image_features = feature_extractor.get_features(query_image)
            heatmap = []
            for template_matcher in template_matchers:
                temp_heatmap, _, _ = template_matcher.get_heatmap(query_image_features)
                heatmap.append(temp_heatmap)
            heatmap = torch.stack(heatmap).mean(0)

            heatmap = np.pad(
                heatmap.cpu().numpy(),
                (
                    (
                        (query_image.shape[0] - heatmap.shape[0]) // 2,
                        (query_image.shape[0] - heatmap.shape[0]) // 2,
                    ),
                    (
                        (query_image.shape[1] - heatmap.shape[1]) // 2,
                        (query_image.shape[1] - heatmap.shape[1]) // 2,
                    ),
                ),
                "constant",
                constant_values=heatmap.cpu().numpy().max(),
            )
            local_max = peak_local_max(
                -heatmap, min_distance=patch_size[0] // 5, exclude_border=False
            )
            local_max_vals = heatmap[local_max[:, 0], local_max[:, 1]]
            possible_bboxes = pd.DataFrame(
                {
                    "x1": local_max[:, 1] - patch_size[0] // 2,
                    "y1": local_max[:, 0] - patch_size[1] // 2,
                    "x2": local_max[:, 1] + patch_size[0] // 2,
                    "y2": local_max[:, 0] + patch_size[1] // 2,
                    "distance": local_max_vals,
                    "image_id": [query_image_id] * len(local_max_vals),
                }
            )

            # # draw rectangles
            # for i in range(len(query_image_df)):
            #     bbox = query_image_df.iloc[i][["x1", "y1", "x2", "y2"]].values
            #     cv2.rectangle(
            #         query_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 128, 0), 3,
            #     )
            # for i in range(len(possible_bboxes)):
            #     bbox = possible_bboxes.iloc[i][["x1", "y1", "x2", "y2"]].values
            #     cv2.rectangle(
            #         query_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (128, 0, 0), 3,
            #     )
            # plt.imshow(query_image)
            # plt.scatter(local_max[:, 1], local_max[:, 0], c="r", s=local_max_vals * 100)
            # plt.show()

            ious, gt_idx = bops.box_iou(
                torch.tensor(possible_bboxes[["x1", "y1", "x2", "y2"]].values),
                torch.tensor(query_image_df[["x1", "y1", "x2", "y2"]].values),
            ).max(dim=1)
            _, gt_idx1 = bops.box_iou(
                torch.tensor(query_image_df[["x1", "y1", "x2", "y2"]].values),
                torch.tensor(possible_bboxes[["x1", "y1", "x2", "y2"]].values),
            ).max(dim=1)

            ious[gt_idx1[gt_idx] != torch.arange(len(gt_idx))] = 0
            tp_col = ious >= 0.5
            fp_col = ious < 0.5
            # fn_col = ious == 0
            possible_bboxes["iou"] = ious.cpu().numpy()
            possible_bboxes["tp"] = tp_col.cpu().numpy().astype(int)
            possible_bboxes["fp"] = fp_col.cpu().numpy().astype(int)
            all_predictions_df.append(possible_bboxes)

        all_predictions_df = pd.concat(all_predictions_df).reset_index(drop=True)
        all_predictions_df = all_predictions_df.sort_values("distance", ascending=True).reset_index(
            drop=True
        )
        all_predictions_df["dist_percentile"] = all_predictions_df["distance"].rank(pct=True).values
        all_predictions_df["tp_cumsum"] = all_predictions_df["tp"].cumsum()
        all_predictions_df["fp_cumsum"] = all_predictions_df["fp"].cumsum()
        all_predictions_df["fn_cumsum"] = query_images_df.shape[0] - all_predictions_df["tp_cumsum"]
        all_predictions_df["precision"] = all_predictions_df["tp_cumsum"] / (
            all_predictions_df["tp_cumsum"] + all_predictions_df["fp_cumsum"]
        )
        all_predictions_df["recall"] = all_predictions_df["tp_cumsum"] / (
            all_predictions_df["tp_cumsum"] + all_predictions_df["fn_cumsum"]
        )

        intervals = pd.cut(all_predictions_df["recall"], np.linspace(0, 1, 1001))
        all_predictions_df["recall1"] = intervals.apply(lambda x: x.mid)
        pr_df[f"precision_{i}"] = (
            all_predictions_df.groupby("recall1")[["precision"]].mean().reset_index()
        )["precision"].values

        mean_precision = pr_df[[i for i in pr_df.keys() if "precision_" in i]].mean(axis=1).values
        pr_df["mean_precision"] = mean_precision
        pr_df.to_csv(f"{exp_folder}_pr_df.csv", index=False)

        pr_df1 = pr_df.dropna().sort_values("recall_bins", ascending=False).reset_index(drop=True)
        auc = np.trapz(pr_df1["mean_precision"], pr_df1["recall_bins"])
        auc_desc.set_description(f"ImageID: {query_image_id} AUC: {auc:.3f}")

        recall_bins_repeat = np.repeat(pr_df["recall_bins"].values, (i / n_samples) + 1)
        precision_repeat = pr_df[[i for i in pr_df.keys() if "precision_" in i]].values.flatten()
        pr_df2 = pd.DataFrame({"Recall": recall_bins_repeat, "Precision": precision_repeat})
        intervals = pd.cut(pr_df2["Recall"], np.linspace(0, 1, 100))
        pr_df2["Recall"] = intervals.apply(lambda x: x.mid)

        plt.figure()
        sns.lineplot(data=pr_df2, x="Recall", y="Precision")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.savefig(f"{exp_folder}_pr_curve.png")
        plt.close()

        tqdm_dataset.update(n_samples)

    tqdm_dataset.close()

    x = 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_features", type=int, default=64)
    argparser.add_argument("--n_clusters", type=int, default=32)

    # args = argparser.parse_args()
    # main(args.n_clusters, args.n_features)

    datasets_filepath = ["dataset_annotations/cilia_dataset.csv"]

    # datasets_filepath = ["dataset_annotations/tiny_tlp_dataset.csv"]

    models = ["resnet18"]
    n_features = [512]
    n_clusters = [64]  # [32, 64, 128]
    n_haar_features = [3]
    scales = [2]
    n_samples = [1, 2, 3]
    random_seeds = [0, 1, 2, 3, 4]

    exp_folder = "exps/cilia"

    for i, dataset_filepath in enumerate(datasets_filepath):
        dataset_name = dataset_filepath.split("/")[-1].split(".")[0]

        all_mious = []
        all_success_rates = []
        all_features = []
        all_clusters = []
        all_haar_features = []
        all_scales = []
        all_models = []

        for model in models:
            for n_feature in n_features:
                for n_cluster in n_clusters:
                    for scale in scales:
                        for n_haar_feature in n_haar_features:
                            for random_seed in random_seeds:
                                for n_sample in n_samples:
                                    exp_name = f"{dataset_name}_{model}_n_features_{n_feature}_n_cluster_{n_cluster}_n_haar_feature_{n_haar_feature}_scale_{scale}_n_samples_{n_sample}_random_seed_{random_seed}"
                                    set_random_seed(random_seed)
                                    print(f"Running {exp_name}")
                                    main(
                                        os.path.join(exp_folder, exp_name),
                                        dataset_filepath,
                                        model,
                                        n_feature,
                                        n_cluster,
                                        n_haar_feature,
                                        scale,
                                        n_sample,
                                    )

