import time
from ast import literal_eval

import pandas as pd
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from opt import *

import numpy as np


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



def make_sequence_dataset(mode='train',dataset_name='ADL'):

    #val is the same as test
    if mode=='all':
        par_video_id_list=id
    elif mode=='train':
        par_video_id_list = train_video_id
    else:
        par_video_id_list=test_video_id


    print(f'start load {mode} data, #videos: {len(par_video_id_list)}')
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

            if not annos.empty:


                annos_subset=annos[['img_path',
                                                 'nao_bbox_resized', 'label','previous_frames','frame']]
                df_items = df_items.append(annos_subset)



    df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')
    return df_items

# def make_sequence_dataset(mode='train',dataset_name='ADL'):
#
#     #val is the same as test
#     if mode=='all':
#         par_video_id_list=id
#     elif mode=='train':
#         par_video_id_list = train_video_id
#     else:
#         par_video_id_list=test_video_id
#
#
#     print(f'start load {mode} data, #videos: {len(par_video_id_list)}')
#     df_items = pd.DataFrame()
#     for video_id in sorted(par_video_id_list):
#         anno_name = 'nao_' + video_id + '.csv'
#         anno_path = os.path.join(args.data_path, annos_path, anno_name)
#         if os.path.exists(anno_path):
#             # start = time.process_time()
#             if dataset_name=='ADL':
#                 #for ADL dataset, the format of video_id is : P_01
#                 img_path = os.path.join(args.data_path, frames_path,video_id)
#             else:
#                 #for Epic dataset, the format of video_id is: P01P01_01, video_id[0:3] is participant_id,
#                 #and video_id[3:] is the real video_id
#                 img_path=os.path.join(args.data_path,frames_path,video_id[0:3],video_id[3:])
#
#             annos = pd.read_csv(anno_path,
#                                 converters={"nao_bbox": literal_eval,
#                                             "nao_bbox_resized": literal_eval,
#                                             "previous_frames":literal_eval})
#             annos['img_path']=img_path
#
#             if not annos.empty:
#
#
#                 annos_subset=annos[['img_path',
#                                                  'nao_bbox_resized', 'label','previous_frames','frame']]
#                 df_items = df_items.append(annos_subset)
#
#
#
#     df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
#     print('finished')
#     print('=============================================================')
#     return df_items


class NAODataset(Dataset):
    def __init__(self,data, mode='train'):
        self.mode=mode
        self.crop = transforms.RandomCrop((args.img_resize[0],
                                           args.img_resize[1]))
        self.transform_label = transforms.ToTensor()

        # self.data = make_sequence_dataset(mode,dataset_name)
        self.data = data
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([  # [h, w]
            transforms.Resize(args.img_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
            , AddGaussianNoise(0., 0.5)
        ])
        self.transform_test = transforms.Compose([  # [h, w]
            transforms.Resize(args.img_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
        ])

        self.transform_previous_frames = transforms.Compose([  # [h, w]
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
            , AddGaussianNoise(0., 0.5)
        ])
        self.transform_previous_frames_test = transforms.Compose([  # [h, w]
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
        ])

    def __getitem__(self, item):
        rand_num=torch.rand(1) if self.mode=='train' else 0
        # rand_num=0
        df_item = self.data.iloc[item, :]
        nao_bbox = df_item.nao_bbox
        # print(f'original bbox: {nao_bbox}')

        # path where images are stored
        img_dir = df_item.img_path
        previous_frames=[]
        # for i in range(0,1):
        for i in range(0,len(df_item.previous_frames)):
            image_name=f'frame_{str(df_item.previous_frames[i]).zfill(10)}.jpg'
            img=Image.open(os.path.join(img_dir,image_name))
            if rand_num > 0.5:
                img = ImageOps.mirror(img)
            img=self.transform_previous_frames(img) if self.mode=='train' else self.transform_previous_frames_test(img)
            previous_frames.append(img)
            del img
        previous_frames=torch.stack(previous_frames)
        previous_frames=previous_frames.transpose(0,1)
        current_frame_path=os.path.join(img_dir,f'frame_{str(df_item.frame).zfill(10)}.jpg')
        current_frame=Image.open(current_frame_path)
        if rand_num>0.5:
            current_frame = ImageOps.mirror(current_frame)
            temp=nao_bbox[0]
            nao_bbox[0]=455-nao_bbox[2]
            nao_bbox[2] = 455 - temp

        # print(f'new bbox: {nao_bbox}')

        current_frame_tensor=self.transform(current_frame) if self.mode=='train' else self.transform_test(current_frame)
        del current_frame


        return previous_frames,current_frame_tensor, torch.tensor(nao_bbox),current_frame_path

    def __len__(self):
        return self.data.shape[0]




def ini_datasets(dataset_name='ADL',original_split=False):
    if original_split==False:
        data = make_sequence_dataset('all', dataset_name)
        data=data.iloc[np.random.RandomState(seed=args.seed).permutation(len(data))]
        if dataset_name=='ADL':
            # train_data,val_data=data.iloc[0:1767],data.iloc[1767:]
            train_data,val_data=data.iloc[0:1767],data.iloc[1767:]
        else:
            train_data,val_data=data.iloc[0:8589],data.iloc[8589:]
    else:
        train_data, val_data = make_sequence_dataset('train', dataset_name),make_sequence_dataset('val', dataset_name)

    return NAODataset(mode='train', data=train_data), NAODataset(mode='val', data=val_data)


if __name__ == '__main__':
    train_dataset,val_dataset=ini_datasets(dataset_name='ADL',original_split=False)
    # train_dataset = NAODataset(mode='train',data=train_data)
    print(train_dataset.data.head())
    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/EPIC/test.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                                  num_workers=8, shuffle=False,pin_memory=True)
    print(f'start traversing the dataloader')
    start = time.time()
    for epoch in range(10):

        for data in train_dataloader:
            previous_frames,current_frame,nao_bbox,current_frame_path=data
            # print(f'previous frames shape: {previous_frames.shape}')
            # print(f'current_frame shape: {current_frame.shape}')
            print(f'sample frame path: {current_frame_path[0]}, nao_bbox: {nao_bbox[0]}')

            nao_bbox_shape=nao_bbox.shape

            """"
            test data and annotations
            need to undo-resize first !!!
            """
            current_frame_example=current_frame[0].permute(1,2,0).numpy()
            current_frame_example*=255
            cv2.imwrite('test.jpg',current_frame_example)
            cv2_image=cv2.imread('test.jpg')
            cv2_image=cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGB)
            nao_bbox_example=nao_bbox[0]
            cv2.rectangle(cv2_image,(nao_bbox_example[0],nao_bbox_example[1]),(nao_bbox_example[2],nao_bbox_example[3]),(255,0,0),3)
            #
            cv2.imshow('example',cv2_image)
            cv2.waitKey(0)

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
