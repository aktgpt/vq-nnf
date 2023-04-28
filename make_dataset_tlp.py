import glob
import os
import time
from copy import deepcopy
from typing import List

import colorcet as cc
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import torch
from feature_extraction import PixelFeatureExtractor
from matchers.template_matching2 import TemplateMatcher
import numpy as np
import pandas as pd

dataset_folder = "/mnt/hdd1/users/aktgpt/datasets/template_matching/TinyTLP"
video_clips_path = sorted(glob.glob(os.path.join(dataset_folder, "*")))

all_template_paths = []
all_template_annotations = []
all_querie_paths = []
all_querie_annotations = []

for video_clip_path in video_clips_path:
    gt_path = os.path.join(video_clip_path, "groundtruth_rect.txt")
    imgs_path = sorted(glob.glob(os.path.join(video_clip_path, "img", "*.jpg")))
    annotations = pd.read_csv(gt_path, header=None)
    for i in range(0, 500, 10):
        template_path = imgs_path[i]
        template_bbox = annotations.iloc[i, 1:5].values
        query_path = imgs_path[i + 100]
        query_bbox = annotations.iloc[i + 100, 1:5].values
        all_template_paths.append(template_path)
        all_template_annotations.append(template_bbox)
        all_querie_paths.append(query_path)
        all_querie_annotations.append(query_bbox)

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
    }
)
df.to_csv("tiny_tlp_dataset.csv", index=False)

x = 1
