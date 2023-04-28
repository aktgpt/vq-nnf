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
import torch
from torch import nn

from feature_extraction import PixelFeatureExtractor
from matchers.template_matching2 import TemplateMatcher

random.seed(42)

dataset_folder = "/mnt/hdd1/users/aktgpt/datasets/template_matching/BBS"
video_clips_path = sorted(glob.glob(os.path.join(dataset_folder, "*/")))
query_timestep = 100

props_df = pd.read_csv(
    os.path.join(dataset_folder, "bbs_props.csv"),
    header=None,
    names=["seq", "props"],
    dtype={"seq": str, "props": str},
)
props_df_one_hot = pd.get_dummies(props_df["props"].str.split(", ", expand=True).stack()).sum(
    level=0
)
props_df_one_hot["seq"] = props_df["seq"]

random_seeds = [42, 69, 420, 666, 1337]
df_columns = [
    "template_path",
    "template_x",
    "template_y",
    "template_w",
    "template_h",
    "query_path",
    "query_x",
    "query_y",
    "query_w",
    "query_h",
]
df_columns.extend(props_df_one_hot.columns[:-1].tolist())

for iter in range(5):
    np.random.seed(random_seeds[iter])

    dataset_df = pd.DataFrame(columns=df_columns)

    for video_clip_path in video_clips_path:
        seq_name = video_clip_path.split("/")[-2]
        seq_props = props_df_one_hot[props_df_one_hot["seq"] == seq_name]
        gt_path = os.path.join(video_clip_path, "groundtruth_rect.txt")
        imgs_path = sorted(glob.glob(os.path.join(video_clip_path, "img", "*.jpg")))

        annotations = pd.read_csv(gt_path, header=None)
        if annotations.iloc[:, :].values.dtype != int:
            annotations = pd.read_csv(gt_path, header=None, sep="\t")

        n_samples = annotations.shape[0] - query_timestep
        try:
            random_samples = np.random.choice(n_samples, 3, replace=True)
            for idx in random_samples:
                template_path = imgs_path[idx]
                template_bbox = annotations.iloc[idx, :].values
                query_path = imgs_path[idx + query_timestep]
                query_bbox = annotations.iloc[idx + query_timestep, :].values

                dataset_df = dataset_df.append(
                    {
                        "template_path": template_path,
                        "template_x": template_bbox[0],
                        "template_y": template_bbox[1],
                        "template_w": template_bbox[2],
                        "template_h": template_bbox[3],
                        "query_path": query_path,
                        "query_x": query_bbox[0],
                        "query_y": query_bbox[1],
                        "query_w": query_bbox[2],
                        "query_h": query_bbox[3],
                        "IV": seq_props["IV"].values[0],
                        "SV": seq_props["SV"].values[0],
                        "OCC": seq_props["OCC"].values[0],
                        "DEF": seq_props["DEF"].values[0],
                        "MB": seq_props["MB"].values[0],
                        "FM": seq_props["FM"].values[0],
                        "IPR": seq_props["IPR"].values[0],
                        "OPR": seq_props["OPR"].values[0],
                        "OV": seq_props["OV"].values[0],
                        "BC": seq_props["BC"].values[0],
                        "LR": seq_props["LR"].values[0],
                    },
                    ignore_index=True,
                )
        except:
            print(video_clip_path, n_samples)

    dataset_df.to_csv(
        f"dataset_annotations/bbs{query_timestep}_iter{iter+1}_dataset.csv", index=False
    )

x = 1
