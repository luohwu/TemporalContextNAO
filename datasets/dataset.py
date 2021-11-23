import math
import os
import time
import pickle
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from opt import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.io import read_image,ImageReadMode
import  cv2



def generate_pseudo_track_id(annos):
    video_id = annos.id[0]
    annos.loc[:, 'pseudo_track_id'] = -1
    track_id_ = 0
    for label in annos.label.unique():
        anno_ = annos[annos['label'] == label]
        if anno_.shape[0] <= 3:
            # print(label)
            annos.loc[annos['label'] == label, 'pseudo_track_id'] = \
                video_id + '_' + str(track_id_).zfill(3)
            track_id_ += 1
        else:
            # print(f'{label}: {anno_.shape[0]}')
            # frame_1 = anno_.iloc[0, 0]
            for j, frame in enumerate(anno_.frame):
                if j == 0:
                    annos.loc[anno_.index[0], 'pseudo_track_id'] = \
                        video_id + '_' + str(track_id_).zfill(3)
                else:
                    if (frame - anno_.iloc[j - 1, 0]) < 90:
                        annos.loc[anno_.index[j], 'pseudo_track_id'] = \
                            video_id + '_' + str(track_id_).zfill(3)
                    else:
                        track_id_ += 1
                        annos.loc[anno_.index[j], 'pseudo_track_id'] = \
                            video_id + '_' + str(track_id_).zfill(3)
            track_id_ += 1


def check_pseudo_track_id(annos):
    video_id = annos.id[0]
    annos.loc[:, 'pseudo_track_id'] = -1
    track_id_ = 0
    for label in annos.label.unique():
        anno_ = annos[annos['label'] == label]
        if anno_.shape[0] <= 3:
            # print(label)
            annos.loc[annos['label'] == label, 'pseudo_track_id'] = track_id_
            track_id_ += 1
        else:
            print(f'{label}: {anno_.shape[0]}')
            # frame_1 = anno_.iloc[0, 0]
            for j, frame in enumerate(sorted(anno_.frame)):
                if j == 0:
                    annos.loc[anno_.index[0], 'pseudo_track_id'] = track_id_
                else:
                    if (frame - anno_.iloc[j - 1, 0]) < 90:
                        annos.loc[anno_.index[j], 'pseudo_track_id'] = track_id_
                    else:
                        track_id_ += 1
                        annos.loc[anno_.index[j], 'pseudo_track_id'] = track_id_
            track_id_ += 1


def check_data_annos(args):
    df_items = pd.DataFrame(columns=['img_file', 'pseudo_track_id',
                                     'nao_bbox', 'label'])

    for video_id in sorted(train_video_id):
        start = time.process_time()
        img_path = os.path.join(args.data_path, frames_path,
                                str(video_id)[:3], str(video_id)[3:])

        anno_name = 'nao_' + video_id + '.csv'
        anno_path = os.path.join(args.data_path, annos_path, anno_name)
        annos = pd.read_csv(anno_path, converters={"nao_bbox": literal_eval})

        check_pseudo_track_id(annos)  # 生成track_id

        annos.insert(loc=5, column='img_file', value=0)
        for index in annos.index:
            img_file = img_path + '/' + str(annos.loc[index, 'frame']).zfill(
                10) + '.jpg'
            annos.loc[index, 'img_file'] = img_file

        annos_df = pd.DataFrame(annos, columns=['img_file', 'pseudo_track_id',
                                                'nao_bbox', 'label'])
        df_items = df_items.append(annos_df, ignore_index=False)

        end = time.process_time()
        print(f'finished video {video_id}, time is {end - start}')

    # 生成sequence data
    for idx, pt_id in enumerate(sorted(df_items.pseudo_track_id.unique())):
        df_items.loc[df_items.pseudo_track_id == pt_id, 'bs_idx'] = str(idx)

    print('================================================================')
    return df_items

def combine_all_frames(row):
    previous_frames=row['previous_frames']
    previous_frames.append(row['frame'])
    # return (row['previous_frames']).append(row['frame'])
    # return (row['previous_frames'])
    return previous_frames

def make_sequence_dataset(mode='train',dataset_name='ADL'):

    #val is the same as test
    par_video_id_list=train_video_id if mode=='train' else test_video_id

    print(f'start load {mode} data, size: {len(par_video_id_list)}')
    df_items = pd.DataFrame()
    for video_id in sorted(par_video_id_list):
        anno_name = 'nao_' + video_id + '.csv'
        anno_path = os.path.join(args.data_path, annos_path, anno_name)
        if os.path.exists(anno_path):
            # start = time.process_time()
            if dataset_name=='ADL':
                #for ADL dataset, the format of video_id is : P_01
                img_path = os.path.join(args.data_path, frames_path,video_id)
            else:
                #for Epic dataset, the format of video_id is: P01P01_01, video_id[0:3] is participant_id,
                #and video_id[3:] is the real video_id
                img_path=os.path.join(args.data_path,frames_path,video_id[0:3],video_id[3:])

            annos = pd.read_csv(anno_path,
                                converters={"nao_bbox": literal_eval,
                                            "nao_bbox_resized": literal_eval,
                                            "previous_frames":literal_eval})
            annos['img_path']=img_path
            annos['all_frames']=annos.apply(combine_all_frames,axis=1)

            if not annos.empty:
                generate_pseudo_track_id(annos)  # 生成track_id


                annos_subset=annos[['img_path',  'pseudo_track_id',
                                                 'nao_bbox_resized', 'label','all_frames']]
                df_items = df_items.append(annos_subset)


    for idx, pt_id in enumerate(sorted(df_items.pseudo_track_id.unique())):
        df_items.loc[df_items.pseudo_track_id == pt_id, 'bs_idx'] = str(idx)

    df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')
    return df_items



class NAODataset(Dataset):
    def __init__(self, mode='train',dataset_name='ADL'):
        self.args = args
        self.crop = transforms.RandomCrop((args.img_resize[0],
                                           args.img_resize[1]))
        self.transform_label = transforms.ToTensor()

        self.data = make_sequence_dataset(mode,dataset_name)
        self.dataset_name=dataset_name


        # print(f'{mode} data: {self.data.shape[0]}')

        # # pandas的shuffle
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if args.normalize:
            self.transform = transforms.Compose([  # [h, w]
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # ImageNet
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, item):
        df_item = self.data.iloc[item, :]

        # path where images are stored
        img_dir = df_item.img_path
        all_frames=df_item.all_frames
        images=[]
        # for i in range(0,1):
        for i in range(0,len(all_frames)):
            image_name=f'frame_{str(all_frames[i]).zfill(10)}.jpg'
            # img_io=read_image(os.path.join(img_dir,image_name),ImageReadMode.RGB)
            # img_io=img_io/255.
            # img_io=self.Normalize(img_io)
            img=Image.open(os.path.join(img_dir,image_name))
            # print(img)
            img=img.resize((self.args.img_resize[1],
                          self.args.img_resize[0]))
            img=self.transform(img)
            # print(f'io: {img_io.shape}, original: {img.shape}')
            # print(torch.eq(img_io,img))
            # print(img)
            # print(img_io)
            images.append(img)
        images=torch.cat(images,dim=0)

        return images, torch.tensor(df_item.nao_bbox)

    def __len__(self):  # batch迭代的次数与其有关
        return self.data.shape[0]






if __name__ == '__main__':
    # check_data_annos(args)
    # train_dataset = EpicDataset(args)
    train_dataset = NAODataset(mode='train',dataset_name=args.dataset)
    train_dataset.data.to_csv('/media/luohwu/T7/dataset/EPIC/test.csv',index=False)
    train_dataloader = DataLoader(train_dataset, batch_size=4,
                                  num_workers=8, shuffle=False,pin_memory=True)
    print(f'start traversing the dataloader')
    start = time.time()
    for data in train_dataloader:
        images,nao_bbox=data
        images_shape=images.shape
        nao_bbox_shape=nao_bbox.shape
    end = time.time()
    print(f'used time: {end-start}')
        # print(f'images size: {images.shape}, nao_bbox: {nao_bbox.shape}')
    # print(len(train_dataloader.dataset),train_dataset.__len__())
    # # for data in train_dataloader:
    # it=iter(train_dataloader)
    # nao_bbox_list=[]
    # img,nao_bbox,hand_hm=next(it)
    # nao_bbox_list.append(nao_bbox)
    # nao_bbox_list.append(nao_bbox)
    # print(nao_bbox.shape)
    # nao_bbox_total=torch.cat(nao_bbox_list,0)
    # print(nao_bbox_total.shape)


    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/ADL/test.csv',index=False)
    # for i in range(100):
    #     img, mask, hand_hm = train_dataset.__getitem__(i)
    #     hand_hm=hand_hm.squeeze(0)
    #     img_numpy=img.numpy().transpose(1,2,0)
    #     cv2.imshow('image',img_numpy)
    #     cv2.imshow('image_mask', mask.numpy())
    #     cv2.imshow('hand_mask', hand_hm.numpy())
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     print(img.shape)
    #     print(mask.shape)
    #     print(hand_hm.shape)
    # train_dataset.generate_img_mask_pair()
    # train_dataset.generate_hm()
    # train_dataset = EpicSequenceDataset(args)
    # train_dataloader = DataLoader(train_dataset)
    # train_dataloader = DataLoader(train_dataset, batch_size=4,
    #                               num_workers=3, shuffle=False)
    # # sequence_lens = []
    # for i, data in enumerate(train_dataloader):
    #     img, mask, hand_hm = data
    #     # sequence_lens.append(img.shape[0])
    #     # show(img, mask)
    #     # print(img.shape)
