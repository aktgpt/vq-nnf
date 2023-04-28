import argparse
import os
import random
import shutil

import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import colorcet as cc
import seaborn as sns

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTHONHASHSEED"] = str(42)

import torch
import torchvision.ops.boxes as bops

from feature_extraction import PixelFeatureExtractor
from matchers.template_matching5 import HaarTemplateMatcher

# matplotlib.use("Agg")


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


def main(
    exp_folder,
    dataset_filepath,
    model_name,
    n_features,
    n_clusters,
    n_haar_features,
    scale,
    pca_dims,
    params_list,
    verbose=False,
):
    dataset_csv = pd.read_csv(dataset_filepath)

    feature_extractor = PixelFeatureExtractor(model_name=model_name, num_features=n_features)

    ious = []
    mean_ious = []
    success_rates = []
    temp_ws = []
    temp_hs = []
    image_sizes = []
    temp_match_time = []
    mean_temp_match_time = []
    kmeans_time = []
    mean_kmeans_time = []
    start_index = 0

    tqdm_dataset = tqdm(desc="Images Processed", total=len(dataset_csv), position=0)
    iou_desc = tqdm(total=0, position=1, bar_format="{desc}")
    iou_mean_desc = tqdm(total=0, position=2, bar_format="{desc}")
    success_rate_desc = tqdm(total=0, position=3, bar_format="{desc}")
    time_desc = tqdm(total=0, position=4, bar_format="{desc}")
    kmeans_time_desc = tqdm(total=0, position=5, bar_format="{desc}")

    if verbose:
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
        else:
            shutil.rmtree(exp_folder)
            os.makedirs(exp_folder)

    for i, row in dataset_csv[start_index:].iterrows():
        template_image = cv2.cvtColor(cv2.imread(row["template_path"]), cv2.COLOR_BGR2RGB)
        query_image = cv2.cvtColor(cv2.imread(row["query_path"]), cv2.COLOR_BGR2RGB)
        template_bbox = row[["template_x", "template_y", "template_w", "template_h"]].values
        query_bbox = row[["query_x", "query_y", "query_w", "query_h"]].values

        template_image_features = feature_extractor.get_features(template_image)
        # template_image_features = feature_extractor.get_color_features(template_image)

        temp_x, temp_y, temp_w, temp_h = template_bbox
        temp_ws.append(temp_w)
        temp_hs.append(temp_h)
        image_sizes.append(query_image.shape[0] * query_image.shape[1])
        temp_x = max(temp_x, 0)
        temp_y = max(temp_y, 0)

        template_features = template_image_features[
            :, temp_y : temp_y + temp_h, temp_x : temp_x + temp_w
        ]
        template = template_image[temp_y : temp_y + temp_h, temp_x : temp_x + temp_w]
        template_matcher = HaarTemplateMatcher(
            template_features,
            (temp_h, temp_w),
            n_clusters=n_clusters,
            n_haar_features=n_haar_features,
            scales=scale,
            pca_dims=pca_dims,
            # template_image=template,
            params_weights=params_list,
            verbose=verbose,
        )
        query_image_features = feature_extractor.get_features(query_image)
        # query_image_features = feature_extractor.get_color_features(query_image)

        torch.cuda.synchronize()
        t1 = time.time()
        heatmap, template_labels, labels = template_matcher.get_heatmap(query_image_features)
        torch.cuda.synchronize()
        t2 = time.time()

        heatmap = heatmap.cpu().numpy()

        query_x, query_y = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        query_w, query_h = template_bbox[3], template_bbox[2]

        # fig, ax = plt.subplots(1, 4, figsize=(15, 5))
        # ax[0].imshow(template)
        # ax[1].imshow(template_labels, cmap="jet")
        # rect = plt.Rectangle(
        #     (query_y, query_x), query_h, query_w, linewidth=2, edgecolor="r", facecolor="none"
        # )
        # ax[2].add_patch(rect)
        # ax[2].imshow(query_image)
        # ax[2].imshow(heatmap1, alpha=0.5)
        # ax[3].imshow(labels, cmap="jet")
        # rect = plt.Rectangle(
        #     (query_y, query_x), query_h, query_w, linewidth=2, edgecolor="r", facecolor="none"
        # )
        # ax[3].add_patch(rect)
        # plt.show()

        bbox_iou = bops.box_iou(
            torch.tensor([query_x, query_y, query_x + query_w, query_y + query_h]).unsqueeze(0),
            torch.tensor(
                [
                    query_bbox[1],
                    query_bbox[0],
                    query_bbox[1] + query_bbox[3],
                    query_bbox[0] + query_bbox[2],
                ]
            ).unsqueeze(0),
        )
        ious.append(bbox_iou.item())
        mean_ious.append(np.mean(ious))
        success_rates.append(np.mean(np.array(ious) > 0.5))
        temp_match_time.append(t2 - t1)
        mean_temp_match_time.append(np.mean(temp_match_time))
        kmeans_time.append(template_matcher.kmeans_time)
        mean_kmeans_time.append(np.mean(kmeans_time))

        tqdm_dataset.update(1)
        iou_desc.set_description_str(f"IOU: {bbox_iou.item()}")
        iou_mean_desc.set_description_str(f"IOU Mean: {np.mean(ious)}")
        success_rate_desc.set_description_str(f"Success Rate: {np.mean(np.array(ious) > 0.5)}")
        time_desc.set_description_str(f"Time: {np.mean(temp_match_time)}")
        kmeans_time_desc.set_description_str(f"Kmeans Time: {np.mean(kmeans_time)}")
        if verbose == True:
            heatmap1 = np.pad(
                heatmap,
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
                constant_values=heatmap.min(),
            )
            heatmap1 = cv2.applyColorMap(
                (((heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min())) * 255).astype(
                    np.uint8
                ),
                cv2.COLORMAP_JET,
            )
            cv2.imwrite(
                os.path.join(exp_folder, f"{i}_template.png"),
                cv2.cvtColor(template, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(exp_folder, f"{i}_template_image.png"),
                cv2.rectangle(
                    cv2.cvtColor(template_image, cv2.COLOR_RGB2BGR),
                    (temp_x, temp_y),
                    (temp_x + temp_w, temp_y + temp_h),
                    (0, 255, 0),
                    2,
                ),
            )
            cv2.imwrite(
                os.path.join(exp_folder, f"{i}_template_labels.png"),
                (template_labels * 255).astype(np.uint8),
            )
            cv2.imwrite(
                os.path.join(exp_folder, f"{i}_query_image.png"),
                # cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR),
                cv2.rectangle(
                    cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR),
                    #     cv2.rectangle(
                    #         cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR),
                    #         (query_y, query_x),
                    #         (query_y + query_h, query_x + query_w),
                    #         (255, 255, 0),
                    #         2,  # noqa
                    #     ),
                    (query_bbox[0], query_bbox[1]),
                    (query_bbox[0] + query_bbox[2], query_bbox[1] + query_bbox[3]),
                    (0, 255, 0),
                    2,
                ),
            )
            cv2.imwrite(
                os.path.join(exp_folder, f"{i}_query_image_labels.png"),
                (labels * 255).astype(np.uint8),
            )

            cv2.imwrite(
                os.path.join(exp_folder, f"{i}_query_image_heatmap.png"),
                # heatmap1,
                cv2.rectangle(
                    # heatmap1,
                    cv2.rectangle(
                        heatmap1,
                        (query_y, query_x),
                        (query_y + query_h, query_x + query_w),
                        (255, 0, 255),
                        2,  # noqa
                    ),
                    (query_bbox[0], query_bbox[1]),
                    (query_bbox[0] + query_bbox[2], query_bbox[1] + query_bbox[3]),
                    (0, 255, 0),
                    2,
                ),
            )
        # iou_df = pd.DataFrame(
        #     {
        #         "ious": ious,
        #         "mean_iou": mean_ious,
        #         "success_rate": success_rates,
        #         # "temp_w": temp_ws,
        #         # "temp_h": temp_hs,
        #         "img_size": image_sizes,
        #         "temp_size": np.array(temp_ws) * np.array(temp_hs),
        #         "time": temp_match_time,
        #         "kmeans_time": kmeans_time,
        #     }
        # )
        # iou_df.to_csv(os.path.join(exp_folder + "_iou_sr.csv"), index=False)
        # iou_df = pd.DataFrame(
        #     {
        #         "ious": ious,
        #         "mean_iou": mean_ious,
        #         "success_rate": success_rates,
        #         # "temp_w": temp_ws,
        #         # "temp_h": temp_hs,
        #         "img_size": image_sizes,
        #         "temp_size": np.array(temp_ws) * np.array(temp_hs),
        #         "time": temp_match_time,
        #     }
        # )
        # iou_df.to_csv(os.path.join(exp_folder + "_iou_sr.csv"), index=False)
    # iou_df[["BC", "DEF", "FM", "IPR", "IV", "LR", "MB", "OCC", "OPR", "OV", "SV"]] = dataset_csv[
    #     ["BC", "DEF", "FM", "IPR", "IV", "LR", "MB", "OCC", "OPR", "OV", "SV"]
    # ]
    # iou_df.to_csv(os.path.join(exp_folder + "_iou_sr.csv"), index=False)
    return mean_ious[-1], success_rates[-1], mean_temp_match_time[-1], mean_kmeans_time[-1]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--n_features", type=int, default=64)
    argparser.add_argument("--n_clusters", type=int, default=32)

    # args = argparser.parse_args()
    # main(args.n_clusters, args.n_features)

    # datasets_filepath = [f"dataset_annotations/bbs100_iter{i}_dataset.csv" for i in range(1, 6)]
    datasets_filepath = ["dataset_annotations/tiny_tlp_dataset.csv"]
    # datasets_filepath = ["dataset_annotations/tlpattr_dataset.csv"]

    models = ["resnet18"]  # ["efficientnet-b0"]
    # "resnet50", "resnet34",
    n_features = [512]  #  [40, 80, 192, 512]
    n_clusters = [128]  # [40, 80, 192, 512
    n_haar_features = [2]  # [1, 3, 5,]
    scales = [2]  # 1, 2, 3,
    pca_dims = [None, 18, 9]

    # params_1x = [1]  # [1.0]  # [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    # params_2x = [1]  # [0.0, 0.25, 0.5, 1.0, 2.0]
    # params_3x = [1]  # [0.0, 0.25, 0.5, 1.0, 2.0]
    # params_4x = [1]  # [0.0, 0.25, 0.5, 1.0, 2.0]

    # exp_folder = "exps_final/bbs100/pca"
    exp_folder = "exps_final/tiny_tlp/pca"
    # exp_folder = "exps_final/tlpattr/pca"

    for i, dataset_filepath in enumerate(datasets_filepath):
        dataset_name = dataset_filepath.split("/")[-1].split(".")[0]

        # if os.path.exists(os.path.join(exp_folder, f"{dataset_name}_all_time_results.csv")):
        #     df = pd.read_csv(os.path.join(exp_folder, f"{dataset_name}_all_time_results.csv"))
        #     all_mious = df["M_IOU"].values.tolist()
        #     all_success_rates = df["Success_Rate"].values.tolist()
        #     all_temp_match_time = df["Temp_Match_Time"].values.tolist()
        #     all_features = df["n_features"].values.tolist()
        #     all_clusters = df["n_clusters"].values.tolist()
        #     all_haar_features = df["n_haar_features"].values.tolist()
        #     all_scales = df["scales"].values.tolist()
        #     all_models = df["model"].values.tolist()
        # else:

        all_mious = []
        all_success_rates = []
        all_temp_match_time = []
        all_kmeans_time = []
        all_features = []
        all_clusters = []
        all_haar_features = []
        all_scales = []
        all_models = []
        all_pca_dims = []
        # all_params_1x = []
        # all_params_2x = []
        # all_params_3x = []
        # all_params_4x = []

        for model in models:
            for n_feature in n_features:
                for n_cluster in n_clusters:
                    for scale in scales:
                        for n_haar_feature in n_haar_features:
                            for pca_dim in pca_dims:
                                # for param_1x in params_1x:
                                #     for param_2x in params_2x:
                                #         for param_3x in params_3x:
                                #             for param_4x in params_4x:
                                exp_name = f"{dataset_name}_{model}_n_features_{n_feature}_n_cluster_{n_cluster}_n_haar_feature_{n_haar_feature}_scale_{scale}"  # _1x_{param_1x}_2x_{param_2x}_3x_{param_3x}_4x_{param_4x}"
                                print(f"Running {exp_name}")
                                mious, success_rates, temp_match_time, kmeans_time = main(
                                    os.path.join(exp_folder, exp_name),  # "rel_diff",
                                    dataset_filepath,
                                    model,
                                    n_feature,
                                    n_cluster,
                                    n_haar_feature,
                                    scale,
                                    pca_dim,
                                    None,  # [param_1x, param_2x, param_3x, param_4x],
                                    verbose=False,
                                )
                                all_mious.append(mious)
                                all_success_rates.append(success_rates)
                                all_temp_match_time.append(temp_match_time)
                                all_kmeans_time.append(kmeans_time)
                                all_features.append(n_feature)
                                all_clusters.append(n_cluster)
                                all_haar_features.append(n_haar_feature)
                                all_scales.append(scale)
                                all_models.append(model)
                                all_pca_dims.append(pca_dim)
                                # all_params_1x.append(param_1x)
                                # all_params_2x.append(param_2x)
                                # all_params_3x.append(param_3x)
                                # all_params_4x.append(param_4x)
                                print(f"Finished {exp_name}")

                                df = pd.DataFrame(
                                    {
                                        "model": all_models,
                                        "n_features": all_features,
                                        "n_clusters": all_clusters,
                                        "scales": all_scales,
                                        "n_haar_features": all_haar_features,
                                        "pca_dim": all_pca_dims,
                                        # "param_1x": all_params_1x,
                                        # "param_2x": all_params_2x,
                                        # "param_3x": all_params_3x,
                                        # "param_4x": all_params_4x,
                                        "M_IOU": all_mious,
                                        "Success_Rate": all_success_rates,
                                        "Temp_Match_Time": all_temp_match_time,
                                        "Kmeans_Time": all_kmeans_time,
                                    }
                                )
                                df.to_csv(
                                    os.path.join(
                                        exp_folder,
                                        # f"{dataset_name}_all_time_results_rel_diff.csv"
                                        f"{dataset_name}_all_time_results_k3s2haar_pca.csv"
                                        # f"{dataset_name}_eff_all_time_results.csv"
                                        # f"{dataset_name}_params_explore_results.csv",
                                    ),
                                    index=False,
                                )

