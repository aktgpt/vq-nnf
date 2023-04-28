import argparse
import os
import random
import time

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTHONHASHSEED"] = str(42)

import matplotlib.patches as patches
import torch
import torchvision.ops.boxes as bops
from skimage.feature import peak_local_max
from sklearn import metrics

from feature_extraction import PixelFeatureExtractor
from matchers.template_matching5 import HaarTemplateMatcher

matplotlib.use("TkAgg")
sns.set_style("whitegrid")
sns.set_context("notebook")  # paper, notebook, talk, poster


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


image_size = (512, 512)  # (512, 512)
patch_size = (64, 64)  # (64, 64)
min_val, max_val = np.array([12, 3, 8]), np.array([255, 255, 255])
class_ids = [1]


def get_bbox(image_shape, x1, y1, x2, y2):
    scale_x = image_shape[0] / image_size[0]
    scale_y = image_shape[1] / image_size[0]
    x_center = int((x1 + x2) / (2 * scale_x))
    y_center = int((y1 + y2) / (2 * scale_y))
    x1 = int(x_center - (patch_size[0] / 2))
    y1 = int(y_center - (patch_size[1] / 2))
    x2 = x1 + patch_size[0]
    # y2 = int(y_center + (self.patch_size[1] / 2))
    y2 = y1 + patch_size[1]
    return np.array([y1, x1, y2, x2])


def get_min_max_percentile(df, percentile=0.5):
    images = []
    for i, row in df.iterrows():
        image = cv2.imread(
            os.path.join(
                "/mnt/hdd1/users/aktgpt/datasets/template_matching/malaria/images", row["image_id"],
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

    # all_images = np.unique(dataset_csv["image_id"])
    # images_for_exp = random.sample(all_images.tolist(), 200)
    # dataset_csv = dataset_csv[dataset_csv["image_id"].isin(images_for_exp)].reset_index(drop=True)
    # dataset_csv.to_csv(f"{exp_folder}_dataset.csv")

    feature_extractor = PixelFeatureExtractor(model_name=model_name, num_features=n_features)

    for class_id in class_ids:
        dataset_csv_class = dataset_csv.iloc[
            np.where(dataset_csv["class_id"] == class_id)
        ].reset_index(drop=True)
        dataset_csv_class["freq"] = len(dataset_csv_class) / dataset_csv_class.groupby("image_id")[
            "image_id"
        ].transform("count")
        sampled_df = dataset_csv_class.sample(n=50 * n_samples, weights="freq").reset_index(
            drop=True
        )
        sampled_df.to_csv(f"{exp_folder}_class_id_{class_id}_samples.csv")

        tqdm_dataset = tqdm(desc="Images Processed", total=len(sampled_df), position=0)
        auc_desc = tqdm(total=0, position=1, bar_format="{desc}")
        template_box_desc = tqdm(total=0, position=2, bar_format="{desc}")

        pr_df = pd.DataFrame({"recall_bins": np.round(np.arange(0.0005, 1.0, 0.001), 4)})
        for i in range(0, len(sampled_df), n_samples):
            rows = sampled_df.iloc[i : i + n_samples]
            template_matchers = []
            query_images = unique_images.copy()
            for _, row in rows.iterrows():
                image_id = row["image_id"]
                image = cv2.cvtColor(
                    cv2.imread(
                        os.path.join(
                            "/mnt/hdd1/users/aktgpt/datasets/template_matching/malaria/images",
                            image_id,
                        ),
                        -1,
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                img_shape = image.shape[:2]
                image = cv2.resize(
                    (normalize_image(image, min_val, max_val) * 255).astype(np.uint8), image_size,
                )
                image_features = feature_extractor.get_features(image)

                template_bbox = get_bbox(img_shape, row["x1"], row["y1"], row["x2"], row["y2"])
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

            query_images_df = dataset_csv_class[dataset_csv_class["image_id"].isin(query_images)]

            query_progress = tqdm(
                total=len(query_images), position=3, desc="Query Images Processed",
            )
            query_desc = tqdm(total=len(query_images), position=4, bar_format="{desc}")

            all_predictions_df = []
            for query_image_id in query_images:
                query_image_df = dataset_csv_class[dataset_csv_class["image_id"] == query_image_id]
                query_image = cv2.cvtColor(
                    cv2.imread(
                        os.path.join(
                            "/mnt/hdd1/users/aktgpt/datasets/template_matching/malaria/images",
                            query_image_id,
                        ),
                        -1,
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                query_shape = query_image.shape[:2]
                query_image = cv2.resize(
                    (normalize_image(query_image, min_val, max_val) * 255).astype(np.uint8),
                    image_size,
                )
                query_image_features = feature_extractor.get_features(query_image)
                heatmap = []
                for template_matcher in template_matchers:
                    temp_heatmap, _, _ = template_matcher.get_heatmap(query_image_features)
                    heatmap.append(temp_heatmap)
                heatmap = torch.stack(heatmap).mean(dim=0).cpu().numpy()
                # heatmap = torch.stack(heatmap).max(dim=0)[0].cpu().numpy()
                # heatmap = np.pad(
                #     heatmap.cpu().numpy(),
                #     (
                #         (
                #             (image_size[0] - heatmap.shape[0]) // 2,
                #             (image_size[0] - heatmap.shape[0]) // 2,
                #         ),
                #         (
                #             (image_size[1] - heatmap.shape[1]) // 2,
                #             (image_size[1] - heatmap.shape[1]) // 2,
                #         ),
                #     ),
                #     "constant",
                #     constant_values=heatmap.cpu().numpy().max(),
                # )
                
                local_max_vals = heatmap[local_max[:, 0], local_max[:, 1]]
                possible_bboxes = pd.DataFrame(
                    {
                        "x1": local_max[:, 1],  # - patch_size[0] // 2,
                        "y1": local_max[:, 0],  # - patch_size[1] // 2,
                        "x2": local_max[:, 1] + patch_size[0],  # // 2,
                        "y2": local_max[:, 0] + patch_size[1],  # // 2,
                        "distance": local_max_vals,
                        "image_id": [query_image_id] * len(local_max_vals),
                    }
                )
                query_image_bboxes = []
                for _, row in query_image_df.iterrows():
                    query_image_bboxes.append(
                        get_bbox(query_shape, row["x1"], row["y1"], row["x2"], row["y2"])
                    )
                query_image_bboxes = np.stack(query_image_bboxes)
                # # draw rectangles
                # for i in range(len(query_image_bboxes)):
                #     bbox = query_image_bboxes[i]
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
                    torch.tensor(query_image_bboxes),
                ).max(dim=1)
                _, gt_idx1 = bops.box_iou(
                    torch.tensor(query_image_bboxes),
                    torch.tensor(possible_bboxes[["x1", "y1", "x2", "y2"]].values),
                ).max(dim=1)

                ious[gt_idx1[gt_idx] != torch.arange(len(gt_idx))] = 0
                tp_col = ious > 0.5
                fp_col = ious <= 0.5
                possible_bboxes["tp"] = tp_col.cpu().numpy().astype(int)
                possible_bboxes["fp"] = fp_col.cpu().numpy().astype(int)
                possible_bboxes["iou"] = ious.cpu().numpy()
                all_predictions_df.append(possible_bboxes)

                query_progress.update(1)
                query_desc.set_description(
                    f"Query Image: {query_image_id} | TP: {tp_col.sum()} | FP: {fp_col.sum()}"
                )

            all_predictions_df = (
                pd.concat(all_predictions_df)
                .sort_values("distance", ascending=True)
                .reset_index(drop=True)
            )
            all_predictions_df["dist_percentile"] = (
                all_predictions_df["distance"].rank(pct=True).values
            )
            all_predictions_df["tp_cumsum"] = all_predictions_df["tp"].cumsum()
            all_predictions_df["fp_cumsum"] = all_predictions_df["fp"].cumsum()
            all_predictions_df["fn_cumsum"] = (
                query_images_df.shape[0] - all_predictions_df["tp_cumsum"]
            )
            all_predictions_df["precision"] = all_predictions_df["tp_cumsum"] / (
                all_predictions_df["tp_cumsum"] + all_predictions_df["fp_cumsum"]
            )
            all_predictions_df["recall"] = all_predictions_df["tp_cumsum"] / (
                all_predictions_df["tp_cumsum"] + all_predictions_df["fn_cumsum"]
            )

            intervals = pd.cut(all_predictions_df["recall"], np.linspace(0, 1, 1001))
            all_predictions_df["recall1"] = intervals.apply(lambda x: x.mid)
            pr_df[f"precision_{i}"] = (
                all_predictions_df.groupby("recall1")[["precision"]].mean().reset_index(drop=True)
            )["precision"].values
            mean_precision = (
                pr_df[[i for i in pr_df.columns if "precision_" in i]].mean(axis=1).values
            )
            pr_df["mean_precision"] = mean_precision
            pr_df.to_csv(f"{exp_folder}_class_id_{class_id}_pr_df.csv", index=False)

            pr_df1 = (
                pr_df.dropna().sort_values("recall_bins", ascending=True).reset_index(drop=True)
            )
            auc = np.trapz(pr_df1["mean_precision"], pr_df1["recall_bins"])
            auc_desc.set_description(f"ImageID: {query_image_id} AUC: {auc:.4f}")

            recall_bins_repeat = np.repeat(pr_df["recall_bins"].values, (i / n_samples) + 1)
            precision_repeat = pr_df[
                [i for i in pr_df.columns if "precision_" in i]
            ].values.flatten()
            pr_df2 = pd.DataFrame({"Recall": recall_bins_repeat, "Precision": precision_repeat})
            intervals = pd.cut(pr_df2["Recall"], np.linspace(0, 1, 100))
            pr_df2["Recall"] = intervals.apply(lambda x: x.mid)
            pr_df2 = pr_df2.dropna().sort_values("Recall", ascending=True).reset_index(drop=True)

            plt.figure()
            sns.lineplot(data=pr_df2, x="Recall", y="Precision")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.xticks(np.arange(0, 1.1, 0.1))
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.savefig(f"{exp_folder}_class_id_{class_id}_pr_curve.png")
            plt.close()

            tqdm_dataset.update(n_samples)
    tqdm_dataset.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_features", type=int, default=64)
    argparser.add_argument("--n_clusters", type=int, default=32)

    # args = argparser.parse_args()
    # main(args.n_clusters, args.n_features)

    datasets_filepath = ["dataset_annotations/malaria_dataset.csv"]

    # datasets_filepath = ["dataset_annotations/tiny_tlp_dataset.csv"]

    models = ["resnet18"]
    n_features = [512]
    n_clusters = [128]  # [32, 64, 128]
    n_haar_features = [1]
    scales = [1]
    n_samples = [5]
    random_seeds = [0, 1, 2]

    exp_folder = "exps_final/malaria"

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
