# coding:utf-8

import os
import cv2
import tqdm
train_on_gpu = True

import numpy as np
import pandas as pd
import pickle as plk

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import segmentation_models_pytorch as smp
import ttach as tta

from utils import *


test_imgs_folder = "./data/Clouds_Classify/test_images/"
train_imgs_folder = "./data/Clouds_Classify/train_images/"



def main(traing=True):
    """
    5折验证（已经训练好5个模型），logits平均
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

        get_val_logits(valid_ids, i, 'se_resnext101_32x4d', 'unet') # or 'fpn'
        validation(valid_ids, i)



def get_val_logits(valid_ids, num_split, encoder, decoder):
    # valid
    train = "./data/Clouds_Classify/train.csv"

    # Data overview
    train = pd.read_csv(open(train))
    train.head()

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    ENCODER = encoder
    ENCODER_WEIGHTS = 'imagenet'

    if decoder == 'unet':
        #建立模型
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
    valid_bs = 1
    valid_dataset = CloudDataset(df=train, transforms = get_validation_augmentation(), datatype='valid', img_ids=valid_ids, preprocessing=get_preprocessing(preprocessing_fn))
    valid_loader = DataLoader(valid_dataset, batch_size=valid_bs, shuffle=False, num_workers=num_workers)    

    loaders = {"valid": valid_loader}
    logdir = "./logs/log_{}_{}/log_{}".format(encoder, decoder, num_split)

    print('predicting for validation data...')
    ###############使用pytorch TTA预测####################
    use_TTA = True
    checkpoint_path = logdir + '/checkpoints/best.pth'
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
        for _, data in enumerate(tqdm.tqdm(loaders['valid'])):
            img, _, img_name = data
            img = img.cuda()
            batch_preds = tta_model(img).cpu().numpy()
            batch_preds = batch_preds.astype(np.float16)
            
            save_dir = './logits/valid/' + encoder + '_' + decoder + '/log_{}'.format(num_split)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            file_name = img_name[0].split('.')[0] + '.plk'
            file_path = os.path.join(save_dir, file_name)

            with open(file_path, 'wb') as wf:
                plk.dump(batch_preds, wf)



def validation(valid_ids, num_split):
    #读入logits
    ENCODERS = ['se_resnext101_32x4d']
    DECODERS = ['unet', 'fpn']
    logdit_dir = './logits/valid/'
    runner_out = []

    logit_ids = os.listdir(os.path.join(logdit_dir, ENCODERS[0] + '_' + DECODERS[0] + '/log_{}'.format(num_split)))
    print('img_ids:', len(logit_ids))

    print('loading and averaging ...')
    for i,logit_id in enumerate(tqdm.tqdm(logit_ids)):
        logits = []
        for en in ENCODERS:
            for de in  DECODERS:
                logit_path = logdit_dir + en + '_' + de + '/log_{}/'.format(num_split) + logit_id

                with open(logit_path, 'rb') as rf:
                    logit = plk.load(rf)

                logits.append(logit)

        logits_mean = np.mean(np.array(logits), axis=0)
        logits_mean = logits_mean.astype(np.float32) #opencv 需要np.float32

        runner_out.extend(logits_mean)

    runner_out = np.array(runner_out)
    print('runner out:', runner_out.shape)

    # valid
    train = "./data/Clouds_Classify/train.csv"

    # Data overview
    train = pd.read_csv(open(train))

    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    preprocessing_fn = smp.encoders.get_preprocessing_fn('resnet34', 'imagenet')
    valid_dataset = CloudDataset(df=train, transforms = get_validation_augmentation(), datatype='valid', img_ids=valid_ids, preprocessing=get_preprocessing(preprocessing_fn))
    
    #处理mask和预测结果
    print('resizing mask and logits')
    valid_masks = []
    probabilities = np.zeros((len(valid_ids) * 4, 350, 525))
    for i, ((_, mask, _), output) in enumerate(tqdm.tqdm(zip(valid_dataset, runner_out))):
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
    params_1 = [[35, 76], [12000, 18001]]
    params_2 = [[35, 76], [12000, 18001]]
    params_3 = [[35, 76], [8000, 15001]]
    param = [params_0, params_1, params_2, params_3]

    best_param = []
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
        attempts_df.to_csv('./params/logits_ensemble/params_{}/tta_params_{}.csv'.format(num_split, class_id), columns=['threshold', 'size', 'dice'], index=False)

        best = attempts_df.iloc[0].values.tolist()
        print(best_param)
        best_param.append(best)
    print('best params:', str(best_param))



def get_test_logits(encoder, decoder):
    '''
    预测并保存测试集logits
    '''
    sub = "./data/Clouds_Classify/sample_submission.csv"
    sub = pd.read_csv(open(sub))
    
    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
   
    #建立模型
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
    
    #载入模型参数
    logdir = "./logs/log_{}_{}/log_{}".format(encoder, 'fpn', 4)
    checkpoint_path = logdir + '/checkpoints/best.pth'
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    
    #使用tta预测
    use_TTA = True
    if use_TTA:
        print('using TTA!!!')
        transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Scale(scales=[5/6, 1, 7/6]),
            ])
        tta_model = tta.SegmentationTTAWrapper(model, transforms, merge_mode='mean')
    else:
        tta_model = model

    test_ids = [id for id in os.listdir(test_imgs_folder)]

    test_dataset = CloudDataset(df=sub, transforms = get_validation_augmentation(), datatype='test', img_ids=test_ids, preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    tta_model = tta_model.cuda()
    tta_model.eval() 

    with torch.no_grad(): 
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            img, _, img_name = data
            img = img.cuda()
            batch_preds = tta_model(img).cpu().numpy()
            batch_preds = batch_preds.astype(np.float16)


            save_dir = './logits/test/' + encoder + '_fpn' + '/log_{}'.format(4)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            #保存为pickle
            file_name = img_name[0].split('.')[0] + '.plk'
            file_path = os.path.join(save_dir, file_name)

            with open(file_path, 'wb') as wf:
                plk.dump(batch_preds, wf)


def testing(num_split, class_params):
    """
    读取不同模型logits，模型平均并测试推理
    """
    ENCODERS = ['se_resnext101_32x4d']
    DECODERS = ['unet', 'fpn']
    logdit_dir = './logits/test/'

    logit_ids = os.listdir(os.path.join(logdit_dir, ENCODERS[0] + '_' + DECODERS[0] + '/log_{}'.format(num_split)))
    print('img_ids:', len(logit_ids))

    sub = "./data/Clouds_Classify/sample_submission.csv"
    sub = pd.read_csv(open(sub))

    print('loading and averaging ...')
    encoded_pixels = []
    for i,logit_id in enumerate(tqdm.tqdm(logit_ids)):
        logits = []
        for en in ENCODERS:
            for de in DECODERS:
                logit_path = logdit_dir + en + '_' + de + '/log_{}/'.format(num_split) + logit_id

                with open(logit_path, 'rb') as rf:
                    logit = plk.load(rf)

                logits.append(logit)

        logits_mean = np.mean(np.array(logits), axis=0)
        logits_mean = logits_mean.astype(np.float32) #opencv 需要np.float32

        for j,pred in enumerate(logits_mean[0]):
            if pred.shape != (350, 525):
                pred = cv2.resize(pred, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)

            predict, num_predict = post_process(sigmoid(pred), class_params[j][0], class_params[j][1])

            if num_predict == 0:
                encoded_pixels.append('')
            else:
                r = mask2rle(predict)
                encoded_pixels.append(r)

    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv('./sub/average/submission_average_unet_fpn_{}.csv'.format(num_split), columns=['Image_Label', 'EncodedPixels'], index=False)




if __name__ == '__main__':
    main()
    # get_test_logits('se_resnext101_32x4d', 'unet')
    # c_1 = {0: [0.4, 16000], 1: [0.6, 14000], 2: [0.5, 18000], 3: [0.35, 12000]}
    # testing(1, c_1)