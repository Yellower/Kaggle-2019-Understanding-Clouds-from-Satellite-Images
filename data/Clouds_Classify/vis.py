# from __future__ import print_function, division
import os
import time
import numpy as np
import torch
import pandas as pd
from skimage import io, transform

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import cv2

from tqdm import tqdm_notebook, tqdm

DIR_TRAIN_IMAGES = "/data/Clouds_Classify/train_images/"

df_train = pd.read_csv(open("/data/Clouds_Classify/train.csv"))
df_train["Label"] = [i.split("_")[1] for i in df_train.Image_Label.values.tolist()]

df_train.head(20)

# print(df_train)

df_train_grp = df_train.dropna().groupby("Label")

idx_fish = df_train_grp.get_group("Fish").index.values.tolist()
idx_flower = df_train_grp.get_group("Flower").index.values.tolist()
idx_gravel = df_train_grp.get_group("Gravel").index.values.tolist()
idx_sugar = df_train_grp.get_group("Sugar").index.values.tolist()
# print(idx_fish, len(idx_flower), len(idx_gravel), len(idx_sugar))

def get_mask(idx:int):
    n = idx
    name_tuple = df_train.iloc[n, 0].split("_")
    img_name, img_label = name_tuple[0], name_tuple[1] 
    encoded_pixels = df_train.iloc[n,1]

    enc_pix =  list(map(int,encoded_pixels.split(" ")))
    print(enc_pix,'111111')
    pix, pix_count = [], []
    for i in tqdm(range(0, len(enc_pix))):
        if i%2==0:
            pix.append(enc_pix[i])
        else:
            pix_count.append(enc_pix[i])

    t1 = time.time()
    rle_pixels = [list(range(pix[i],pix[i]+pix_count[i])) for i in range(0, len(pix))]
    rle_mask_pixels = sum(rle_pixels,[])
    print(time.time()-t1)

    img_loc = DIR_TRAIN_IMAGES+img_name
    img = cv2.imread(img_loc)
    mask_img = np.zeros((img.shape[0]*img.shape[1],1), dtype=int)
    mask_img[rle_mask_pixels] = 255
    height,width=img.shape[0], img.shape[1]
    mask = np.reshape(mask_img, (width, height)).T
    

    return mask, img_label, img

# #plt.imshow(mask);

def disp_image(idx:int):
    mask, img_label, img = get_mask(idx)
    # img_loc = DIR_TRAIN_IMAGES+img_name
    # img = cv2.imread(img_loc)
    # img_to_disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img[mask==0,1] = 255
    return img, img_label


def disp_sample(category:str, ls_idx:list):
    for i in range(len(ls_idx)):
        fig=plt.figure(figsize=(20, 10))
        # fig.subplots_adjust(hspace=0.9, wspace=0.4)

        # ax = fig.add_subplot(1, 1, 1)
        img2disp, _ = disp_image(ls_idx[i])
        # ax.set_title(category)
        plt.imshow(img2disp)

        plt.tight_layout()
        plt.savefig("/data/Clouds_Classify/vis_images/image_{}_{}.png".format(category, ls_idx[i]))
        # plt.show()

disp_sample("fish", idx_fish)
disp_sample("flower", idx_flower)
disp_sample("sugar", idx_sugar)
disp_sample("gravel", idx_gravel)

# def disp_sample_assorted():
#     fig=plt.figure(figsize=(18, 12))
#     #fig.subplots_adjust(hspace=0.9, wspace=0.4)
#     columns = 3
#     rows = 3
#     for i in range(1,10):
#         ax = fig.add_subplot(rows, columns, i)
#         img2disp, img_label = disp_image(random.choice(idx_fish+idx_flower+idx_sugar+idx_gravel))
#         ax.set_title(img_label)
#         plt.imshow(img2disp)

#     plt.tight_layout()
#     plt.show()

# disp_sample_assorted()

