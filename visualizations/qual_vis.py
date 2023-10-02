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
from itertools import combinations, product
from scipy import stats
import cv2

sns.set_style("whitegrid")
sns.set_context("talk")  # paper, notebook, talk, poster
folder = "exps_final"
plt.rcParams["font.family"] = "Arial"


def read_gt(file_path):
    with open(file_path) as IN:
        x, y, w, h = [eval(i) for i in IN.readline().strip().split(",")]
    return x, y, w, h


methods = ["DDIS", "DIWU", "DeepDIM_2_19_25", "VQ_NNF"]
# result_folders = ["c:/Users/gupta/Desktop/TLPattr/TLPattr_comp_scale_1.00_TM_Results"]

result_folders = [f"C:/Users/gupta/Desktop/BBS_data/BBS100_iter{i}_TM_Results" for i in range(1, 6)]
result_folders.extend([f"C:/Users/gupta/Desktop/BBS_data/BBS50_iter{i}_TM_Results" for i in range(1, 6)])
result_folders.extend([f"C:/Users/gupta/Desktop/BBS_data/BBS100_iter{i}_TM_Results" for i in range(1, 6)])


for result_folder in result_folders:
    save_folder = f"{result_folder}/VQ_NNF/method_comparison"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    ddis_df = pd.read_csv(f"{result_folder}/DDIS/DDIS_iter_512_results.csv")
    diwu_df = pd.read_csv(f"{result_folder}/DIWU/DIWU_iter_512_results.csv")
    deepdim_df = pd.read_csv(f"{result_folder}/DeepDIM_2_19_25/iou_sr.csv")
    vqnnf_df = pd.read_csv(
        f"{result_folder}/VQ_NNF/model_resnet18_n_feats_512_n_codes_128_haar_filts_2_scale_2/iou_sr.csv"
    )

    dataset_folder = result_folder.replace("_TM_Results", "")

    bboxes_path = sorted([os.path.join(dataset_folder, i) for i in os.listdir(dataset_folder) if ".txt" in i])
    imgs_path = sorted([os.path.join(dataset_folder, i) for i in os.listdir(dataset_folder) if ".jpg" in i])

    num_samples = len(bboxes_path) // 2

    for idx in range(num_samples):
        template_image = cv2.cvtColor(cv2.imread(imgs_path[2 * idx]), cv2.COLOR_BGR2RGB)
        query_image = cv2.cvtColor(cv2.imread(imgs_path[2 * idx + 1]), cv2.COLOR_BGR2RGB)
        template_bbox = read_gt(bboxes_path[2 * idx])
        query_gt_bbox = read_gt(bboxes_path[2 * idx + 1])

        cv2.imwrite(
            f"{save_folder}/{idx}_template.png",
            cv2.rectangle(
                cv2.cvtColor(template_image, cv2.COLOR_RGB2BGR),
                (template_bbox[0], template_bbox[1]),
                (template_bbox[0] + template_bbox[2], template_bbox[1] + template_bbox[3]),
                (0, 255, 0),
                3,
            ),
        )

        query_image = cv2.rectangle(
            cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR),
            (query_gt_bbox[0], query_gt_bbox[1]),
            (query_gt_bbox[0] + query_gt_bbox[2], query_gt_bbox[1] + query_gt_bbox[3]),
            (0, 255, 0),
            3,
        )

        ## ddis box
        ddis_bbox = np.array(
            [ddis_df.iloc[idx]["x"], ddis_df.iloc[idx]["y"], ddis_df.iloc[idx]["w"], ddis_df.iloc[idx]["h"]]
        ).astype(int)
        query_image = cv2.rectangle(
            query_image,
            (ddis_bbox[0], ddis_bbox[1]),
            (ddis_bbox[0] + ddis_bbox[2], ddis_bbox[1] + ddis_bbox[3]),
            (255, 0, 255),
            3,
        )
        ## diwu box
        diwu_bbox = np.array(
            [diwu_df.iloc[idx]["x"], diwu_df.iloc[idx]["y"], diwu_df.iloc[idx]["w"], diwu_df.iloc[idx]["h"]]
        ).astype(int)
        query_image = cv2.rectangle(
            query_image,
            (diwu_bbox[0], diwu_bbox[1]),
            (diwu_bbox[0] + diwu_bbox[2], diwu_bbox[1] + diwu_bbox[3]),
            (0, 0, 255),
            3,
        )
        ## deepdim box
        deepdim_bbox = np.array(
            [deepdim_df.iloc[idx]["x"], deepdim_df.iloc[idx]["y"], deepdim_df.iloc[idx]["w"], deepdim_df.iloc[idx]["h"]]
        ).astype(int)
        query_image = cv2.rectangle(
            query_image,
            (deepdim_bbox[0], deepdim_bbox[1]),
            (deepdim_bbox[0] + deepdim_bbox[2], deepdim_bbox[1] + deepdim_bbox[3]),
            (255, 255, 0),
            3,
        )
        ## vqnnf box
        vqnnf_bbox = np.array(
            [vqnnf_df.iloc[idx]["x"], vqnnf_df.iloc[idx]["y"], vqnnf_df.iloc[idx]["w"], vqnnf_df.iloc[idx]["h"]]
        ).astype(int)
        query_image = cv2.rectangle(
            query_image,
            (vqnnf_bbox[0], vqnnf_bbox[1]),
            (vqnnf_bbox[0] + vqnnf_bbox[2], vqnnf_bbox[1] + vqnnf_bbox[3]),
            (0, 165, 255),
            3,
        )
        cv2.imwrite(f"{save_folder}/{idx}_query.png", query_image)

        ddis_heatmap = cv2.imread(f"{result_folder}/DDIS/seperated/{idx+1}/DDIS Deep_map.jpg")

        cv2.imwrite(
            f"{save_folder}/{idx}_ddis_heatmap.png",
            cv2.rectangle(
                cv2.rectangle(
                    ddis_heatmap,
                    (ddis_bbox[0], ddis_bbox[1]),
                    (ddis_bbox[0] + ddis_bbox[2], ddis_bbox[1] + ddis_bbox[3]),
                    (255, 0, 255),
                    3,
                ),
                (query_gt_bbox[0], query_gt_bbox[1]),
                (query_gt_bbox[0] + query_gt_bbox[2], query_gt_bbox[1] + query_gt_bbox[3]),
                (0, 255, 0),
                3,
            ),
        )

        diwu_heatmap = cv2.imread(f"{result_folder}/DIWU/seperated/{idx+1}/DIWU Deep_map.jpg")
        cv2.imwrite(
            f"{save_folder}/{idx}_diwu_heatmap.png",
            cv2.rectangle(
                cv2.rectangle(
                    diwu_heatmap,
                    (diwu_bbox[0], diwu_bbox[1]),
                    (diwu_bbox[0] + diwu_bbox[2], diwu_bbox[1] + diwu_bbox[3]),
                    (0, 0, 255),
                    3,
                ),
                (query_gt_bbox[0], query_gt_bbox[1]),
                (query_gt_bbox[0] + query_gt_bbox[2], query_gt_bbox[1] + query_gt_bbox[3]),
                (0, 255, 0),
                3,
            ),
        )

        deepdim_heatmap = cv2.imread(f"{result_folder}/DeepDIM_2_19_25/{idx+1}_heatmap.png")

        cv2.imwrite(
            f"{save_folder}/{idx}_deepdim_heatmap.png",
            cv2.rectangle(
                cv2.rectangle(
                    deepdim_heatmap,
                    (deepdim_bbox[0], deepdim_bbox[1]),
                    (deepdim_bbox[0] + deepdim_bbox[2], deepdim_bbox[1] + deepdim_bbox[3]),
                    (255, 255, 0),
                    3,
                ),
                (query_gt_bbox[0], query_gt_bbox[1]),
                (query_gt_bbox[0] + query_gt_bbox[2], query_gt_bbox[1] + query_gt_bbox[3]),
                (0, 255, 0),
                3,
            ),
        )

        vqnnf_heatmap = cv2.imread(
            f"{result_folder}/VQ_NNF/model_resnet18_n_feats_512_n_codes_128_haar_filts_2_scale_2/{idx+1}_heatmap.png"
        )

        cv2.imwrite(
            f"{save_folder}/{idx}_vqnnf_heatmap.png",
            cv2.rectangle(
                cv2.rectangle(
                    vqnnf_heatmap,
                    (vqnnf_bbox[0], vqnnf_bbox[1]),
                    (vqnnf_bbox[0] + vqnnf_bbox[2], vqnnf_bbox[1] + vqnnf_bbox[3]),
                    (0, 165, 255),
                    3,
                ),
                (query_gt_bbox[0], query_gt_bbox[1]),
                (query_gt_bbox[0] + query_gt_bbox[2], query_gt_bbox[1] + query_gt_bbox[3]),
                (0, 255, 0),
                3,
            ),
        )

    x = 1
