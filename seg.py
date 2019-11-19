# coding:utf-8

import os
import cv2
import tqdm
train_on_gpu = True

import numpy as np
import pandas as pd

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, InferCallback, CheckpointCallback
import segmentation_models_pytorch as smp
import ttach as tta

from utils import *

test_imgs_folder = "./data/Clouds_Classify/test_images/"
train_imgs_folder = "./data/Clouds_Classify/train_images/"

def main():
    """
    5折交叉训练和验证
    """
    data_resume = True
    if data_resume:
        with open('./dataset/set_1.txt', 'r') as f1:
            split_1 = [line.strip() for line in f1.readlines()]
        with open('./dataset/set_2.txt', 'r') as f2:
            split_2 = [line.strip() for line in f2.readlines()]
        with open('./dataset/set_3.txt', 'r') as f3:
            split_3 = [line.strip() for line in f3.readlines()]
        with open('./dataset/set_4.txt', 'r') as f4:
            split_4 = [line.strip() for line in f4.readlines()]
        with open('./dataset/set_5.txt', 'r') as f5:
            split_5 = [line.strip() for line in f5.readlines()]
        split_list = [split_1, split_2, split_3, split_4, split_5]
    else:
        # split test and train
        img_ids = [id for id in os.listdir(train_imgs_folder)]
        split_list = split_dataset(img_ids)

    for i in range(5):
        valid_ids = split_list[i]
        train_ids = []
        for j in range(5):
            if j != i:
                train_ids += split_list[j]

        training(train_ids, valid_ids, i, 'se_resnext101_32x4d', 'unet') # or 'fpn'
        validation(valid_ids, i, 'se_resnext101_32x4d', 'unet') # or 'fpn'


def training(train_ids, valid_ids, num_split, encoder, decoder):
    """
    模型训练
    """
    train = "./data/Clouds_Classify/train.csv"

    # Data overview
    train = pd.read_csv(open(train))
    train.head()

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    ENCODER = encoder
    ENCODER_WEIGHTS = 'imagenet'

    if decoder == 'unet':
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=4, 
            activation=None,
        )
    else:
        model = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=4, 
            activation=None,
        )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    num_workers = 4
    bs = 12
    train_dataset = CloudDataset(df=train, transforms = get_training_augmentation(), datatype='train', img_ids=train_ids, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(df=train, transforms = get_validation_augmentation(), datatype='valid', img_ids=valid_ids, preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    num_epochs = 50
    logdir = "./logs/log_{}_{}/log_{}".format(encoder, decoder, num_split)

    # model, criterion, optimizer
    optimizer = torch.optim.Adam([
        {'params': model.decoder.parameters(), 'lr': 1e-2},
        {'params': model.encoder.parameters(), 'lr': 1e-3},
    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.35, patience=4)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    runner = SupervisedRunner()

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback()],
        logdir=logdir,
        num_epochs=num_epochs,
        verbose=True
    )

    # Exploring predictions
    loaders = {"infer": valid_loader}
    runner.infer(
        model=model,
        loaders=loaders,
        callbacks=[
            CheckpointCallback(
                resume=f"{logdir}/checkpoints/best.pth"),
            InferCallback()
        ],
    )


def validation(valid_ids, num_split, encoder, decoder):
    """
    模型验证，并选择后处理参数
    """
    train = "./data/Clouds_Classify/train.csv"

    # Data overview
    train = pd.read_csv(open(train))
    train.head()

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    ENCODER = encoder
    ENCODER_WEIGHTS = 'imagenet'
    if decoder == 'unet':
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=4, 
            activation=None,
        )
    else:
        model = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=4, 
            activation=None,
        )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    num_workers = 4
    valid_bs = 32
    valid_dataset = CloudDataset(df=train, transforms = get_validation_augmentation(), datatype='valid', img_ids=valid_ids, preprocessing=get_preprocessing(preprocessing_fn))
    valid_loader = DataLoader(valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=num_workers)    

    loaders = {"valid": valid_loader}
    logdir = "./logs/log_{}_{}/log_{}".format(encoder, decoder, num_split)
    
    valid_masks = []
    probabilities = np.zeros((len(valid_ids) * 4, 350, 525))

    ############### TTA预测 ####################
    use_TTA = True
    checkpoint_path = logdir + '/checkpoints/best.pth'
    runner_out = []
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

    if use_TTA:
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Scale(scales=[5/6, 1, 7/6]),
        ])
        tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')
    else:
        tta_model = model
    
    tta_model = tta_model.cuda()
    tta_model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(loaders['valid'])):
            img, _ = data
            img = img.cuda()
            batch_preds = tta_model(img).cpu().numpy()
            runner_out.extend(batch_preds)
    runner_out = np.array(runner_out)
    ######################END##########################
    
    for i, ((_, mask), output) in enumerate(tqdm.tqdm(zip(valid_dataset, runner_out))):
        for m in mask:
            if m.shape != (350, 525):
                m = cv2.resize(m, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            probabilities[i * 4 + j, :, :] = probability


    # Find optimal values
    print('searching for optimal param...')
    params_0 = [[35, 76], [12000, 19001]]
    params_1 = [[35, 76], [12000, 19001]]
    params_2 = [[35, 76], [12000, 19001]]
    params_3 = [[35, 76], [8000, 15001]]
    param = [params_0, params_1, params_2, params_3]

    for class_id in range(4):
        par = param[class_id]
        attempts = []
        for t in range(par[0][0], par[0][1], 5):
            t /= 100
            for ms in range(par[1][0], par[1][1], 2000):
                masks = []
                print('==> searching [class_id:%d threshold:%.3f ms:%d]' % (class_id, t, ms))
                for i in tqdm.tqdm(range(class_id, len(probabilities), 4)):
                    probability = probabilities[i]
                    predict, _ = post_process(sigmoid(probability), t, ms)
                    masks.append(predict)

                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))

                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        attempts_df.to_csv('./params/{}_{}_par/params_{}/tta_params_{}.csv'.format(encoder, decoder, num_split, class_id), columns=['threshold', 'size', 'dice'], index=False)


def testing(num_split, class_params, encoder, decoder):
    """
    测试推理
    """
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    sub = "./data/Clouds_Classify/sample_submission.csv"
    sub = pd.read_csv(open(sub))
    sub.head()
    
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, 'imagenet')
    if decoder == 'unet':
        model = smp.Unet(
            encoder_name=encoder, 
            encoder_weights='imagenet', 
            classes=4, 
            activation=None,
        )
    else:
        model = smp.FPN(
            encoder_name=encoder, 
            encoder_weights='imagenet', 
            classes=4, 
            activation=None,
        )
    test_ids = [id for id in os.listdir(test_imgs_folder)]

    test_dataset = CloudDataset(df=sub, transforms = get_validation_augmentation(), datatype='test', img_ids=test_ids, preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    loaders = {"test": test_loader}
    logdir = "./logs/log_{}_{}/log_{}".format(encoder, decoder, num_split)

    encoded_pixels = []

    ###############使用pytorch TTA预测####################
    use_TTA = True
    checkpoint_path = logdir + '/checkpoints/best.pth'
    runner_out = []
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    #使用tta预测
    if use_TTA:
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Scale(scales=[5/6, 1, 7/6]),
        ])
        tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')
    else:
        tta_model = model
    
    tta_model = tta_model.cuda()
    tta_model.eval()

    with torch.no_grad(): 
        for i, data in enumerate(tqdm.tqdm(loaders['test'])):
            img, _ = data
            img = img.cuda()
            batch_preds = tta_model(img).cpu().numpy()
            runner_out.extend(batch_preds)
    runner_out = np.array(runner_out)
    
    
    for i, output in tqdm.tqdm(enumerate(runner_out)):
        for j, probability in enumerate(output):
            if probability.shape != (350, 525):
                probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
            logit = sigmoid(probability)
            predict, num_predict = post_process(logit, class_params[j][0], class_params[j][1])

            if num_predict == 0:
                    encoded_pixels.append('')
            else:
                r = mask2rle(predict)
                encoded_pixels.append(r)

    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv('./sub/{}_{}/tta_submission_{}.csv'.format(encoder, decoder, num_split), columns=['Image_Label', 'EncodedPixels'], index=False)



if __name__ == '__main__':
    main()
    # class_params = [{0: [0.4, 16000], 1: [0.6, 14000], 2: [0.5, 18000], 3: [0.35, 12000]},
    #                 {0: [0.4, 16000], 1: [0.6, 14000], 2: [0.5, 18000], 3: [0.35, 12000]},
    #                 {0: [0.45,18000], 1: [0.55,18000], 2: [0.7,18000], 3: [0.45,12000]},
    #                 {0: [0.65,16000], 1: [0.65,16000], 2: [0.6,18000], 3: [0.4,10000]},
    #                 {0: [0.4,16000], 1: [0.35,16000], 2: [0.6,18000], 3: [0.35,12000]}]
    # for i in range(5):
    #     testing(i, class_params[i], 'se_resnext101_32x4d', 'unet')