# coding:utf-8

import os
import cv2
import tqdm
train_on_gpu = True

import torch
import ttach as tta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import utils


def mask_ensemble(masks):
    """
    mask投票集成，在后处理之后
    masks: list, [mask0, mask1, mask2, ...], mask0: [A, 350, 525, 4] 0/1
    """
    mask_sum = np.zeros(masks[0].shape, dtype=float)
    for mask in masks:
        mask_sum += mask
    ensemble_mask = np.where(mask_sum < len(masks)//2+1, 0, 1)
    return ensemble_mask


def mask_ensemble_csv(csvs):
    sample_sub = '/data/Clouds_Classify/sample_submission.csv'
    sample_sub = pd.read_csv(open(sample_sub))
    sample_sub.head()

    sample_sub['label'] = sample_sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sample_sub['im_id'] = sample_sub['Image_Label'].apply(lambda x: x.split('_')[0])

    image_name_list = np.unique(sample_sub['im_id'].values).tolist()

    sub_list = []
    for i in range(len(csvs)):
        sub = pd.read_csv(open(csvs[i]))
        sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
        sub_list.append(sub)

    encoded_pixels = []
    for index, image_name in enumerate(tqdm.tqdm(image_name_list)):
        # image = utils.get_img(image_name, file_name='test_images')
        mask_sum = np.zeros((350, 525, 4), dtype=np.float32)
        for sub in sub_list:
            mask = utils.make_mask(sub, image_name=image_name, shape=(350, 525)) # [H, W, 4]
            mask_sum += mask
        ensemble_mask = np.where(mask_sum < len(sub_list)//2+1, 0, 1)
        # utils.visualize(image_name, image, ensemble_mask)
        for i in range(4):
            rle = utils.mask2rle(ensemble_mask[:,:,i])
            encoded_pixels.append(rle)

    sample_sub['EncodedPixels'] = encoded_pixels
    sample_sub.to_csv('./sub/tta_ensemble_submission_5unet_3fpn_1resnet34.csv', columns=['Image_Label', 'EncodedPixels'], index=False)


def logit_ensemble(logits, index=1.0):
    """
    预测logit叠加集成，在后处理之前
    logits: list, [logit0, logit1, logit2, ...], logit0: [A, 350, 525, 4] 0-1float
    """
    logit_sum = np.zeros(logits[0].shape, dtype=float)
    for logit in logits:
        logit_sum += logit ** index
    return logit_sum



def main():
    pass


if __name__ == '__main__':
    # csv_0 = './sub/ensemble/tta_ensemble_submission_4unet_4fpn_ensresnet34_6626.csv'
    # csv_1 = './sub/ensemble/tta_ensemble_submission_4unet_5fpn_6621.csv'
    # csv_2 = './sub/ensemble/tta_ensemble_submission_5_unet_4_fpn_6630.csv'
    # csv_3 = './sub/ensemble/tta_ensemble_submission_5_unet_4_fpn_6630.csv'
    # csv_4 = './sub/ensemble/tta_ensemble_submission_5unet_6572_4fpn_6562_6625.csv'
    # csv_5 = './sub/ensemble/tta_ensemble_submission_5unet_6572_4fpn_6637.csv'
    # csv_6 = './sub/ensemble/tta_ensemble_submission_5unet_6572_4fpn_6637.csv'

    csv_0 = './sub/se_resnext101_32x4d_unet/submission_0_6572.csv'
    csv_1 = './sub/se_resnext101_32x4d_unet/tta_submission_1.csv'
    csv_2 = './sub/se_resnext101_32x4d_unet/tta_submission_2.csv'
    csv_3 = './sub/se_resnext101_32x4d_unet/tta_submission_3.csv'
    csv_4 = './sub/se_resnext101_32x4d_unet/tta_submission_4.csv'

    csv_5 = './sub/se_resnext101_32x4d_fpn/tta_submission_0.csv'
    csv_6 = './sub/se_resnext101_32x4d_fpn/tta_submission_1.csv'
    csv_7 = './sub/se_resnext101_32x4d_fpn/tta_submission_2.csv'
    csv_8 = './sub/se_resnext101_32x4d_fpn/tta_submission_3.csv'
    # csv_9 = './sub/se_resnext101_32x4d_fpn/tta_submission_4.csv'

    # csv_10 = './sub/tta_ensemble_submission_5resnet_fpn.csv'

    mask_ensemble_csv([csv_0, csv_1, csv_2, csv_3, csv_4, csv_5, csv_6, csv_7, csv_8])