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

random.seed(42)
np.random.seed(42)


dataset_folder = "/mnt/hdd1/users/aktgpt/datasets/template_matching/TLPattr"
video_clips_path = sorted(glob.glob(os.path.join(dataset_folder, "*/")))

occlusions = [x.split("/")[-2][:2] for x in video_clips_path]
num_files = [len(glob.glob(os.path.join(x, "img", "*.jpg"))) for x in video_clips_path]
df = pd.DataFrame({"occlusions": occlusions, "num_files": num_files, "path": video_clips_path})
df.to_csv(os.path.join(dataset_folder, "occlusions.csv"), index=False)
# df_mean = df.groupby("occlusions").sum()

all_template_paths = []
all_template_annotations = []
all_querie_paths = []
all_querie_annotations = []
all_clip_len = []
all_occ = []
all_clip_ids = []
for video_clip_path in video_clips_path:
    gt_path = os.path.join(video_clip_path, "groundtruth_rect.txt")
    imgs_path = sorted(glob.glob(os.path.join(video_clip_path, "img", "*.jpg")))
    annotations = pd.read_csv(gt_path, header=None)
    for i in range(0, len(annotations) - 100, 10):
        template_path = imgs_path[i]
        template_bbox = annotations.iloc[i, 1:5].values
        query_path = imgs_path[i + 100]
        query_bbox = annotations.iloc[i + 100, 1:5].values
        all_template_paths.append(template_path)
        all_template_annotations.append(template_bbox)
        all_querie_paths.append(query_path)
        all_querie_annotations.append(query_bbox)
        all_occ.append(video_clip_path.split("/")[-2][:2])
        all_clip_ids.append(video_clip_path.split("/")[-2][3:])
all_template_annotations = np.array(all_template_annotations)
all_querie_annotations = np.array(all_querie_annotations)
df = pd.DataFrame(
    {
        "template_path": all_template_paths,
        "template_x": all_template_annotations[:, 0],
        "template_y": all_template_annotations[:, 1],
        "template_w": all_template_annotations[:, 2],
        "template_h": all_template_annotations[:, 3],
        "query_path": all_querie_paths,
        "query_x": all_querie_annotations[:, 0],
        "query_y": all_querie_annotations[:, 1],
        "query_w": all_querie_annotations[:, 2],
        "query_h": all_querie_annotations[:, 3],
        "occlusions": all_occ,
        "clip_id": all_clip_ids,
    }
)
## pick 10 random samples from each occlusion category with weighted sampling based on number of files in each clip_id
df = df.sample(frac=1).reset_index(drop=True)
df1 = df.groupby(["occlusions", "clip_id"]).head(15).reset_index(drop=True)
df1.drop(columns=["clip_id"], inplace=True)
df1.to_csv("dataset_annotations/tlpattr_dataset.csv", index=False)
