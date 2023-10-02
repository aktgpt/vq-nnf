import argparse
import os
import random
import shutil

import time
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTHONHASHSEED"] = str(42)

import torch
import torchvision.ops.boxes as bops

from matching.feature_extraction import PixelFeatureExtractor
from matching.template_matching import VQNNFMatcher

# matplotlib.use("Agg")


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True


def read_gt(file_path):
    with open(file_path) as IN:
        x, y, w, h = [eval(i) for i in IN.readline().strip().split(",")]
    return x, y, w, h


def odd(f):
    return int(np.ceil(f)) // 2 * 2 + 1


def get_rotated_bbox(rotation, image_shape, query_bbox):
    center_query_image = np.array(image_shape[::-1]) / 2
    query_bbox_all_points_centered = np.array(
        [
            query_bbox[:2] - center_query_image,
            query_bbox[:2] + [query_bbox[2], 0] - center_query_image,
            query_bbox[:2] + [0, query_bbox[3]] - center_query_image,
            query_bbox[:2] + [query_bbox[2], query_bbox[3]] - center_query_image,
        ]
    )
    degrees = np.deg2rad(rotation)
    rotation_matrix = np.array(
        [
            [np.cos(degrees), -np.sin(degrees)],
            [np.sin(degrees), np.cos(degrees)],
        ]
    )
    query_bbox_rotated = np.matmul(query_bbox_all_points_centered, rotation_matrix) + center_query_image
    new_query_bbox = (
        np.array([np.min(query_bbox_rotated, axis=0), np.max(query_bbox_rotated, axis=0)]).astype(int).reshape(-1)
    )

    return new_query_bbox


def runTM(
    dataset_folder,
    result_folder,
    model_name,
    n_feature,
    n_code,
    rect_haar_filter,
    scale,
    pca_dim,
    verbose,
    run_config=None,
):
    exp_suffix = f"model_{model_name}_n_feats_{n_feature}_pcadims_{pca_dim}_n_codes_{n_code}_haar_filts_{rect_haar_filter}_scale_{scale}"
    exp_folder = f"{result_folder}/{exp_suffix}"
    print(f"Dataset: {dataset_folder}, Running {exp_suffix}")

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    feature_extractor = PixelFeatureExtractor(model_name=model_name, num_features=n_feature)

    bboxes_path = sorted([os.path.join(dataset_folder, i) for i in os.listdir(dataset_folder) if ".txt" in i])
    imgs_path = sorted([os.path.join(dataset_folder, i) for i in os.listdir(dataset_folder) if ".jpg" in i])
    num_samples = len(imgs_path) // 2

    tqdm_dataset = tqdm(desc="Images Processed", total=num_samples, position=0)
    iou_desc = tqdm(total=0, position=1, bar_format="{desc}")
    iou_mean_desc = tqdm(total=0, position=2, bar_format="{desc}")
    success_rate_desc = tqdm(total=0, position=3, bar_format="{desc}")
    time_desc = tqdm(total=0, position=4, bar_format="{desc}")
    kmeans_time_desc = tqdm(total=0, position=5, bar_format="{desc}")

    ious = []
    temp_ws = []
    temp_hs = []
    image_sizes = []
    temp_match_time = []
    kmeans_time = []
    xs = []
    ys = []
    ws = []
    hs = []

    for idx in range(num_samples):
        template_image = cv2.cvtColor(cv2.imread(imgs_path[2 * idx]), cv2.COLOR_BGR2RGB)
        query_image = cv2.cvtColor(cv2.imread(imgs_path[2 * idx + 1]), cv2.COLOR_BGR2RGB)
        template_bbox = read_gt(bboxes_path[2 * idx])
        query_gt_bbox = read_gt(bboxes_path[2 * idx + 1])

        template_image_features = feature_extractor.get_features(template_image)

        temp_x, temp_y, temp_w, temp_h = template_bbox
        temp_ws.append(temp_w)
        temp_hs.append(temp_h)
        image_sizes.append(query_image.shape[0] * query_image.shape[1])
        temp_x = max(temp_x, 0)
        temp_y = max(temp_y, 0)

        template_features = template_image_features[:, temp_y : temp_y + temp_h, temp_x : temp_x + temp_w]

        template_matcher = VQNNFMatcher(
            template=template_features,
            pca_dims=pca_dim,
            n_code=n_code,
            filters_cat="haar",
            filter_params={"kernel_size": 3, "sigma": 2, "n_scales": scale, "filters": rect_haar_filter},
            verbose=False,
        )

        query_image_features = feature_extractor.get_features(query_image)

        torch.cuda.synchronize()
        t1 = time.time()
        heatmap, filt_heatmaps, template_nnf, query_nnf = template_matcher.get_heatmap(query_image_features)
        torch.cuda.synchronize()
        t2 = time.time()

        # torch.cuda.empty_cache()

        query_w, query_h = template_bbox[3], template_bbox[2]
        if run_config == "rotate":
            angle = int(dataset_folder.split("_rot")[-1]) if "_rot" in dataset_folder else 0
            rotated_bbox = get_rotated_bbox(angle, template_image.shape[:2], np.array(template_bbox))
            query_h, query_w = rotated_bbox[2:] - rotated_bbox[:2]

        query_x, query_y = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        query_x = int(query_x + 1 - (odd(query_w) - 1) / 2)
        query_y = int(query_y + 1 - (odd(query_h) - 1) / 2)

        xs.append(query_y)
        ys.append(query_x)
        ws.append(query_h)
        hs.append(query_w)

        bbox_iou = bops.box_iou(
            torch.tensor([query_x, query_y, query_x + query_w, query_y + query_h]).unsqueeze(0),
            torch.tensor(
                [
                    query_gt_bbox[1],
                    query_gt_bbox[0],
                    query_gt_bbox[1] + query_gt_bbox[3],
                    query_gt_bbox[0] + query_gt_bbox[2],
                ]
            ).unsqueeze(0),
        )

        ious.append(bbox_iou.item())
        temp_match_time.append(t2 - t1)
        kmeans_time.append(template_matcher.kmeans_time)

        tqdm_dataset.update(1)
        iou_desc.set_description_str(f"IOU: {bbox_iou.item()}")
        iou_mean_desc.set_description_str(f"IOU Mean: {np.mean(ious)}")
        success_rate_desc.set_description_str(f"Success Rate: {np.mean(np.array(ious) > 0.5)}")
        time_desc.set_description_str(f"Time: {np.mean(temp_match_time)}")
        kmeans_time_desc.set_description_str(f"Kmeans Time: {np.mean(kmeans_time)}")

        iou_df = pd.DataFrame(
            {
                "ious": ious,
                "x": xs,
                "y": ys,
                "w": ws,
                "h": hs,
                "temp_w": temp_ws,
                "temp_h": temp_hs,
                "img_size": image_sizes,
                "temp_size": np.array(temp_ws) * np.array(temp_hs),
                "time": temp_match_time,
                "kmeans_time": kmeans_time,
            }
        )
        iou_df.to_csv(f"{exp_folder}/iou_sr.csv", index=False)
        if verbose == True:
            cv2.imwrite(
                f"{exp_folder}/{idx+1}_template_image.png",
                cv2.rectangle(
                    cv2.cvtColor(template_image, cv2.COLOR_RGB2BGR),
                    (temp_x, temp_y),
                    (temp_x + temp_w, temp_y + temp_h),
                    (0, 255, 0),
                    3,
                ),
            )

            cv2.imwrite(
                f"{exp_folder}/{idx+1}_query_image.png",
                cv2.rectangle(
                    cv2.rectangle(
                        cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR),
                        (query_y, query_x),
                        (query_y + query_h, query_x + query_w),
                        (255, 0, 255),
                        3,  # noqa
                    ),
                    (query_gt_bbox[0], query_gt_bbox[1]),
                    (
                        query_gt_bbox[0] + query_gt_bbox[2],
                        query_gt_bbox[1] + query_gt_bbox[3],
                    ),
                    (0, 255, 0),
                    3,
                ),
            )

            heatmap = cv2.applyColorMap(
                (((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())) * 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            cv2.imwrite(
                f"{exp_folder}/{idx+1}_heatmap.png",
                cv2.rectangle(
                    heatmap,
                    (query_y, query_x),
                    (query_y + query_h, query_x + query_w),
                    (255, 0, 255),
                    3,  # noqa
                ),
            )

    return np.mean(ious), np.mean(np.array(ious) > 0.5), np.mean(temp_match_time), np.mean(kmeans_time)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--dataset_path", type=str, default="BBS")
    argparser.add_argument("-m", "--model", type=str, default="resnet18")
    argparser.add_argument("-n", "--n_features", type=int, default=512)
    argparser.add_argument("-pc", "--pca_dims", type=int, default=128)
    argparser.add_argument("-c", "--n_codes", type=int, default=128)
    argparser.add_argument("-hf", "--rect_haar_filters", type=int, default=23)
    argparser.add_argument("-s", "--scale", type=int, default=3)
    argparser.add_argument("-r", "--run-config", type=str, default=None)
    argparser.add_argument("-v", "--verbose", action="store_true", default=False)
    # argparser.add_argument("-e", "--exp_suffix", type=str, default="_TM_Results")

    args = argparser.parse_args()

    if os.path.exists(args.dataset_path):
        datasets_folder = [args.dataset_path]
    else:
        if args.dataset_path == "BBS":
            datasets_folder = [f"C:/Users/gupta/Desktop/BBS_data/BBS25_iter{i}" for i in range(1, 6)]
            datasets_folder.extend([f"C:/Users/gupta/Desktop/BBS_data/BBS50_iter{i}" for i in range(1, 6)])
            datasets_folder.extend([f"C:/Users/gupta/Desktop/BBS_data/BBS100_iter{i}" for i in range(1, 6)])

            if args.run_config == "rotate":
                datasets_folder.extend([f"{x}_rot{angle}" for x in datasets_folder for angle in (60, 120, 180)])
                datasets_folder = sorted(datasets_folder)

        elif "TLPattr" in args.dataset_path:
            datasets_folder = [f"c:/Users/gupta/Desktop/TLPattr/TLPattr_comp_scale_1.00"]
            if args.run_config == "scale":
                datasets_folder = [
                    f"c:/Users/gupta/Desktop/TLPattr/TLPattr_comp_scale_{i}"
                    for i in ("0.25", "0.33", "0.50", "0.66", "0.75", "1.00")
                ]

        elif args.dataset_path == "TinyTLP":
            datasets_folder = [f"C:/Users/gupta/Desktop/TinyTLP_comp/TinyTLP_rot0"]
            if args.run_config == "rotate":
                datasets_folder = [
                    f"C:/Users/gupta/Desktop/TinyTLP_comp/TinyTLP_rot{i}" for i in ("0", "60", "120", "180")
                ]

    if args.run_config == "all":
        models = ["resnet18"]  # ["efficientnet-b0"] # "resnet50", "resnet34"
        n_features = [27, 512]  # use 27 for color features
        n_codes = [4, 8, 16, 32, 64, 128]
        rect_haar_filters = [1, 2, 3, 23]
        scales = [1, 2, 3]
    elif args.run_config == "scale":
        models = ["resnet18"]
        n_features = [512, 27]
        n_codes = [128]
        rect_haar_filters = [1, 2, 3, 23]
        scales = [3, 4]
    elif args.run_config == "rotate":
        models = ["resnet18"]
        n_features = [27]
        n_codes = [128]
        rect_haar_filters = [1, 2, 3, 23]
        scales = [4] if args.dataset_path == "TinyTLP" else [3]
    elif args.run_config == "pca":
        models = ["resnet18"]
        n_features = [512]
        n_codes = [128]
        rect_haar_filters = [2] if "BBS" in args.dataset_path else [23]
        scales = [3]
        pca_dims = [None, 18, 9]  # [None, 18, 9]
    else:
        models = [args.model]
        n_features = [args.n_features]
        n_codes = [args.n_codes]
        rect_haar_filters = [args.rect_haar_filters]
        scales = [args.scale]
        pca_dims = [args.pca_dims]

    all_df = pd.DataFrame(
        {
            "dataset": [],
            "model": [],
            "n_features": [],
            "n_codes": [],
            "rect_haar_filters": [],
            "scale": [],
            "pca_dims": [],
            "M_IOU": [],
            "Success_Rate": [],
            "Temp_Match_Time": [],
            "Kmeans_Time": [],
            "Total_Time": [],
        }
    )
    for dataset_folder in datasets_folder:
        for model in models:
            for n_feature in n_features:
                for pca_dim in pca_dims:
                    for n_code in n_codes:
                        for scale in scales:
                            for rect_haar_filter in rect_haar_filters:
                                result_folder = f"{dataset_folder}_TM_Results/VQ_NNF"
                                if not os.path.exists(result_folder):
                                    os.makedirs(result_folder)

                                if args.run_config == "scale":
                                    if n_feature == 27 and scale == 3:
                                        continue
                                    if n_feature == 512 and scale == 4:
                                        continue

                                mean_iou, mean_sr, mean_temp_match_time, mean_kmeans_time = runTM(
                                    dataset_folder=dataset_folder,
                                    result_folder=result_folder,
                                    model_name=model,
                                    n_feature=n_feature,
                                    n_code=n_code,
                                    rect_haar_filter=rect_haar_filter,
                                    scale=scale,
                                    pca_dim=pca_dim,
                                    verbose=args.verbose,
                                    run_config=args.run_config,
                                )

                                dataset_name = dataset_folder.split("/")[-1]
                                all_df = pd.concat(
                                    [
                                        all_df,
                                        pd.DataFrame(
                                            {
                                                "dataset": [dataset_name],
                                                "model": [model],
                                                "n_features": [n_feature],
                                                "n_codes": [n_code],
                                                "scale": [scale],
                                                "rect_haar_filters": [rect_haar_filter],
                                                "pca_dims": [pca_dim],
                                                "M_IOU": [mean_iou],
                                                "Success_Rate": [mean_sr],
                                                "Temp_Match_Time": [mean_temp_match_time],
                                                "Kmeans_Time": [mean_kmeans_time],
                                                "Total_Time": [mean_temp_match_time + mean_kmeans_time],
                                            }
                                        ),
                                    ],
                                    ignore_index=True,
                                )

                                all_df.to_csv(f"{args.dataset_path}_vq_nnf_{args.run_config}_results.csv", index=False)
