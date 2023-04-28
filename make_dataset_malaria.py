import glob
import json
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
import seaborn as sns
import torch
from torch import nn


random.seed(42)

f = open("training.json")
data = json.load(f)
cwd = os.getcwd()

w_s = []
h_s = []
imgs = []
# categories = []

category_count = {
    "image_id": [],
    "difficult": [],
    "leukocyte": [],
    "gametocyte": [],
    "red blood cell": [],
    "ring": [],
    "schizont": [],
    "trophozoite": [],
}

for image_data in data:
    image_path = image_data["image"]["pathname"]
    # imgs.append(os.path.basename(image_path).rsplit(".")[0])
    # image = cv2.imread(cwd + image_path)
    # if os.path.basename(image_path).rsplit(".")[1] == "jpg":

    image_category_count = {
        "leukocyte": 0,
        "difficult": 0,
        "gametocyte": 0,
        "red blood cell": 0,
        "ring": 0,
        "schizont": 0,
        "trophozoite": 0,
    }
    for bbox_data in image_data["objects"]:
        r1, c1 = (
            bbox_data["bounding_box"]["minimum"]["c"],
            bbox_data["bounding_box"]["minimum"]["r"],
        )
        r2, c2 = (
            bbox_data["bounding_box"]["maximum"]["c"],
            bbox_data["bounding_box"]["maximum"]["r"],
        )
        w = r2 - r1
        h = c2 - c1
        w_s.append(w)
        h_s.append(h)
        image_category_count[bbox_data["category"]] += 1
        # categories.append(bbox_data["category"])
        # imgs.append(image_path)
        x = 1

    for key in category_count:
        if key == "image_id":
            category_count["image_id"].append(os.path.basename(image_path).rsplit(".")[0])
        else:
            category_count[key].append(image_category_count[key])


df = pd.DataFrame(category_count)
df["index"] = [i for i in range(len(data))]
# df = df[df.difficult == 0].reset_index(drop=True)
# df = df[df.ring != 0].reset_index(drop=True)

df = df[
    (df.trophozoite > 0)
    & (df.difficult == 0)
    & (df.leukocyte == 0)
    & (df.gametocyte == 0)
    & (df.ring == 0)
    & (df.schizont == 0)
].reset_index(drop=True)
df.to_csv("gt_only_trophozoites.csv", index=False)

dataset_folder = "/mnt/hdd1/users/aktgpt/datasets/template_matching/malaria"
