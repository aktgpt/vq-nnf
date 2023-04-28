import glob
import os
import random
import time
from copy import deepcopy
from typing import List

import colorcet as cc
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shutil
from tqdm import tqdm
from pathlib import Path
import skimage

dataset_folder = "/mnt/hdd1/users/aktgpt/datasets/template_matching/BBS"
data_save_folder = "/mnt/hdd1/users/aktgpt/datasets/template_matching"


def get_rotated_bbox(rotation, image_shape, bbox):
    center_query_image = np.array(image_shape[::-1]) / 2
    bbox_all_points_centered = np.array(
        [
            bbox[:2] - center_query_image,
            bbox[:2] + [bbox[2], 0] - center_query_image,
            bbox[:2] + [0, bbox[3]] - center_query_image,
            bbox[:2] + [bbox[2], bbox[3]] - center_query_image,
        ]
    )
    degrees = np.deg2rad(rotation)
    rotation_matrix = np.array(
        [[np.cos(degrees), -np.sin(degrees)], [np.sin(degrees), np.cos(degrees)],]
    )
    bbox_rotated = np.matmul(bbox_all_points_centered, rotation_matrix) + center_query_image
    new_bbox = np.array(
        [
            np.min(bbox_rotated, axis=0),
            np.max(bbox_rotated, axis=0) - np.min(bbox_rotated, axis=0),
        ],
        dtype=int,
    ).reshape(-1)

    return new_bbox


def make_comp_dataset(timesteps: List[int], iterations: List[int]):
    for timestep in timesteps:
        for i in range(1, iterations + 1):
            pair_idx = 1
            annot_csv = pd.read_csv(f"dataset_annotations/bbs{timestep}_iter{i}_dataset.csv")
            save_folder = os.path.join(data_save_folder, f"BBS{timestep}_iter{i}/")

            if not os.path.exists(save_folder):
                os.makedirs(save_folder, exist_ok=True)
                print(f"Saving to {save_folder}")

            tqdm_iter = tqdm(annot_csv.iterrows(), total=len(annot_csv))
            image_desc = tqdm(total=0, position=1, bar_format="{desc}")
            for j, row in annot_csv.iterrows():
                temp_path = row["template_path"]
                query_path = row["query_path"]
                temp_bbox = [
                    row["template_x"],
                    row["template_y"],
                    row["template_w"],
                    row["template_h"],
                ]
                query_bbox = [row["query_x"], row["query_y"], row["query_w"], row["query_h"]]
                # copy images and rename them
                temp_name = f"pair{str(pair_idx).zfill(4)}_frm1_{temp_path.split('/')[8]}"
                query_name = f"pair{str(pair_idx).zfill(4)}_frm2_{query_path.split('/')[8]}"
                shutil.copy(temp_path, os.path.join(save_folder, temp_name + ".jpg"))
                shutil.copy(query_path, os.path.join(save_folder, query_name + ".jpg"))
                # save annotations with comma separated txt file separately for template and query
                temp_annot = ",".join([str(x) for x in temp_bbox])
                query_annot = ",".join([str(x) for x in query_bbox])
                with open(os.path.join(save_folder, temp_name + ".txt"), "w") as f:
                    f.write(temp_annot)
                with open(os.path.join(save_folder, query_name + ".txt"), "w") as f:
                    f.write(query_annot)
                pair_idx += 1
                tqdm_iter.update(1)
                image_desc.set_description(f"Processed {pair_idx} pairs")

        x = 1


def make_comp_dataset_rotation(timesteps: List[int], iterations: List[int], rotations: List[int]):
    for timestep in timesteps:
        for i in range(1, iterations + 1):
            for rotation in rotations:
                pair_idx = 1
                annot_csv = pd.read_csv(f"dataset_annotations/bbs{timestep}_iter{i}_dataset.csv")
                save_folder = os.path.join(
                    data_save_folder, f"BBS{timestep}_iter{i}_rot{rotation}/"
                )

                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=True)
                    print(f"Saving to {save_folder}")

                tqdm_iter = tqdm(annot_csv.iterrows(), total=len(annot_csv))
                image_desc = tqdm(total=0, position=1, bar_format="{desc}")
                for j, row in annot_csv.iterrows():
                    temp_path = row["template_path"]
                    query_path = row["query_path"]
                    temp_bbox = [
                        row["template_x"],
                        row["template_y"],
                        row["template_w"],
                        row["template_h"],
                    ]
                    query_bbox = row[["query_x", "query_y", "query_w", "query_h"]].values
                    query_image = cv2.imread(query_path, -1)
                    query_image = (skimage.transform.rotate(query_image, rotation) * 255).astype(
                        np.uint8
                    )
                    query_bbox_rotated = get_rotated_bbox(
                        rotation, query_image.shape[:2], query_bbox
                    )

                    temp_name = f"pair{str(pair_idx).zfill(4)}_frm1_{temp_path.split('/')[8]}"
                    query_name = f"pair{str(pair_idx).zfill(4)}_frm2_{query_path.split('/')[8]}"
                    shutil.copy(temp_path, os.path.join(save_folder, temp_name + ".jpg"))
                    cv2.imwrite(os.path.join(save_folder, query_name + ".jpg"), query_image)
                    # cv2.rectangle(
                    #     query_image,
                    #     (query_bbox_rotated[0], query_bbox_rotated[1]),
                    #     (query_bbox_rotated[2], query_bbox_rotated[3],),
                    #     (255, 0, 0),
                    #     2,
                    # )
                    # plt.imshow(query_image)

                    # save annotations with comma separated txt file separately for template and query
                    temp_annot = ",".join([str(x) for x in temp_bbox])
                    query_annot = ",".join([str(x) for x in query_bbox_rotated])
                    with open(os.path.join(save_folder, temp_name + ".txt"), "w") as f:
                        f.write(temp_annot)
                    with open(os.path.join(save_folder, query_name + ".txt"), "w") as f:
                        f.write(query_annot)
                    pair_idx += 1
                    tqdm_iter.update(1)
                    image_desc.set_description(f"Processed {pair_idx} pairs")


iterations = 5
timesteps = [25, 50, 100]
rotations = [60, 120, 180]
# make_comp_dataset(timesteps, iterations)
make_comp_dataset_rotation(timesteps, iterations, rotations)
