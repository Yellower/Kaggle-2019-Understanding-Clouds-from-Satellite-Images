# coding:utf-8

import os
import cv2
import random
train_on_gpu = True

import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import albumentations as albu



def split_dataset(img_ids):
    # 划分数据集
    split_1 = []
    split_2 = []
    split_3 = []
    split_4 = []
    split_5 = []
    for dir in img_ids:
        rand = random.randint(0,100)
        if rand < 20:
            split_1.append(dir)
        elif rand < 40:
            split_2.append(dir)
        elif rand < 60:
            split_3.append(dir)
        elif rand < 80:
            split_4.append(dir)
        else:
            split_5.append(dir)

    with open('./dataset/set_1.txt', 'w') as f1:
        for s in split_1:
            f1.write(s)
            f1.write('\n')
    with open('./dataset/set_2.txt', 'w') as f2:
        for s in split_2:
            f2.write(s)
            f2.write('\n')
    with open('./dataset/set_3.txt', 'w') as f3:
        for s in split_3:
            f3.write(s)
            f3.write('\n')
    with open('./dataset/set_4.txt', 'w') as f4:
        for s in split_4:
            f4.write(s)
            f4.write('\n')
    with open('./dataset/set_5.txt', 'w') as f5:
        for s in split_5:
            f5.write(s)
            f5.write('\n')
    return [split_1, split_2, split_3, split_4, split_5]


def get_img(image_name, file_name='train_images'):
    """
    Return image based on image name and folder.
    """
    train_imgs_folder = "/data/Clouds_Classify/{}/".format(file_name)
    image_path = os.path.join(train_imgs_folder, image_name)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def make_mask(df, image_name='img.jpg', shape=(1400, 2100)):
    """
    Create mask based on df, image name and shape.
    """
    encoded_masks = df.loc[df['im_id'] == image_name, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for idx, label in enumerate(encoded_masks.values):
        if label is not np.nan:
            mask = rle_decode(label, shape=shape)
            masks[:, :, idx] = mask
            
    return masks


def rle_decode(mask_rle='', shape=(1400, 2100)):
    '''
    Decode rle encoded mask.
    
    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def visualize(image_name, image, mask, original_image=None, original_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
    
    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(25, 5))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f'Mask {class_dict[i]}', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig("/data/chencong/cloud_seg/vis_images/ensemble_sub/{}".format(image_name))
    else:
        f, ax = plt.subplots(2, 5, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
                
        for i in range(4):
            ax[0, i + 1].imshow(original_mask[:, :, i])
            ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig("/data/chencong/cloud_seg/vis_images/image.png")
        
        ax[1, 0].imshow(image)
        ax[1, 0].set_title('Transformed image', fontsize=fontsize)
        
        
        for i in range(4):
            ax[1, i + 1].imshow(mask[:, :, i])
            ax[1, i + 1].set_title(f'Transformed mask {class_dict[i]}', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig("/data/chencong/cloud_seg/vis_images/transformed_image.png")


# Setting up data for training in Catalyst
class CloudDataset(Dataset):
    def __init__(self, df, transforms, datatype='train', img_ids=None, preprocessing=None):
        test_imgs_folder = "/data/Clouds_Classify/test_images/"
        train_imgs_folder = "/data/Clouds_Classify/train_images/"
        self.df = df
        if datatype != 'test':
            self.data_folder = train_imgs_folder
        else:
            self.data_folder = test_imgs_folder
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask = make_mask(self.df, image_name)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask, image_name

    def __len__(self):
        return len(self.img_ids)


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=90, p=0.5),
        albu.GridDistortion(p=0.5),
        albu.Resize(384, 576)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(384, 576)
    ]
    return albu.Compose(test_transform)


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    mask = draw_convex_hull(mask.astype(np.uint8))
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def sigmoid(x): 
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def draw_convex_hull(mask, mode='convex'):
    """
    区域凸包
    """
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if mode=='rect': # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)
        elif mode=='convex': # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255),-1)
        elif mode=='approx':
            epsilon = 0.02*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,epsilon,True)
            cv2.drawContours(img, [approx], 0, (255, 255, 255),-1)
        else: # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255),-1)
    return img/255


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


def logit_ensemble(logits, index=1.0):
    """
    预测logit叠加集成，在后处理之前
    logits: list, [logit0, logit1, logit2, ...], logit0: [A, 350, 525, 4] 0-1float
    """
    logit_sum = np.zeros(logits[0].shape, dtype=float)
    for logit in logits:
        logit_sum += logit ** index
    return logit_sum
