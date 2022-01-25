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

def make_sequence_datasetTTC(mode='train',dataset_name='ADL'):
    print(f'dataset name: {dataset_name}')
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
        if dataset_name=='EPIC':
            participant_id=video_id[:3]
            video_id=video_id[3:]
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
                img_path=os.path.join(args.data_path,frames_path,participant_id,video_id)

            annos = pd.read_csv(anno_path,
                                converters={"nao_bbox": literal_eval,
                                            # "nao_bbox_resized": literal_eval,
                                            "previous_frames":literal_eval})
            annos['img_path']=img_path

            if not annos.empty:
                annos_subset = annos[['img_path', 'nao_bbox', 'class', 'previous_frames', 'frame','TTC','TTC_level']]
                df_items = df_items.append(annos_subset)




    # df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    # df_items.to_csv('/media/luohwu/T7/dataset/EPIC/csv_format.csv',index=False)
    print('finished')
    print('=============================================================')
    return df_items


class NAODatasetTTC(Dataset):
    def __init__(self, mode='train',dataset_name='ADL'):
        self.mode=mode
        self.transform_label = transforms.ToTensor()

        self.data = make_sequence_datasetTTC(mode,dataset_name)
        # self.data = data
        # self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([  # [h, w]
            transforms.Resize(args.img_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
            # , AddGaussianNoise(0., 0.5)
        ])
        self.transform_test = transforms.Compose([  # [h, w]
            transforms.Resize(args.img_resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
        ])

        self.transform_previous_frames = transforms.Compose([  # [h, w]
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
            # , AddGaussianNoise(0., 0.5)
        ])
        self.transform_previous_frames_test = transforms.Compose([  # [h, w]
            # transforms.Resize((112,112)) if args.C3D else transforms.Resize((224,224))  ,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])  # ImageNet
        ])

    def __getitem__(self, item):
        # rand_num=torch.rand(1) if self.mode=='train' else 0
        # rand_num=0
        df_item = self.data.iloc[item, :]
        nao_bbox = df_item.nao_bbox
        TTC_level=df_item.TTC_level
        # print(f'original bbox: {nao_bbox}')

        # path where images are stored
        img_dir = df_item.img_path
        frames=[]
        for i in range(0,len(df_item.previous_frames)):
            image_name=f'frame_{str(df_item.previous_frames[i]).zfill(10)}.jpg'
            img=Image.open(os.path.join(img_dir,image_name))
            # if rand_num > 0.5:
            #     img = ImageOps.mirror(img)
            img=self.transform_previous_frames(img)
            frames.append(img)
            del img
        current_frame_path=os.path.join(img_dir,f'frame_{str(df_item.frame).zfill(10)}.jpg')
        current_frame=Image.open(current_frame_path)
        # if rand_num>0.5:
        #     current_frame = ImageOps.mirror(current_frame)
        #     temp=nao_bbox[0]
        #     nao_bbox[0]=455-nao_bbox[2]
        #     nao_bbox[2] = 455 - temp

        # print(f'new bbox: {nao_bbox}')

        current_frame_tensor=self.transform(current_frame)
        frames.append(current_frame_tensor)
        # print(f'shape of current frame: {current_frame_tensor.shape}')
        del current_frame

        frames_tensor=torch.stack(frames)
        del frames
        # print(f'shape of frames{frames_tensor.shape}')


        return frames_tensor, torch.tensor(nao_bbox),current_frame_path,TTC_level
        # return 1, current_frame_tensor, torch.tensor(nao_bbox), current_frame_path

    def __len__(self):
        return self.data.shape[0]

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # train_dataset,val_dataset=ini_datasets(dataset_name='ADL',original_split=False)
    train_dataset = NAODatasetTTC(mode='train',dataset_name=args.dataset)

    print(train_dataset.data.head())
    for i in range(len(train_dataset.data)):
        item=train_dataset.data.iloc[i]
        # if item['TTC_level']>2:
        #     print(item['img_path'][-25:])
        #     print(item['frame'])
        #     print(item['TTC_level'])
    TTC=train_dataset.data['TTC_level']
    print(TTC.shape)
    figure=plt.figure(figsize=(12,8),dpi=1200)
    # plt.plot(range(len(TTC)),TTC,'o')
    plt.hist(TTC,bins=range(1,21))
    plt.xticks(range(1,21,1))
    # plt.show()
    plt.savefig('/media/luohwu/T7/dataset/EPIC/plot.svg')
    cv2.waitKey(0)

    # train_dataset.data.to_csv('/media/luohwu/T7/dataset/EPIC/test.csv')
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                                  num_workers=8, shuffle=False,pin_memory=True)
    print(f'start traversing the dataloader')
    start = time.time()
    for epoch in range(1):

        for data in train_dataloader:
            frames, nao_bbox, current_frame_path,TTC = data
            # print(f'previous frames shape: {previous_frames.shape}')
            # print(f'current_frame shape: {current_frame.shape}')
            # print(f'sample frame path: {current_frame_path[0]}, nao_bbox: {nao_bbox[0]}')
            print(f'current frame path: {current_frame_path[0]}')
            window_name = current_frame_path[0][-25:]

            nao_bbox_shape = nao_bbox.shape

            """"
            test data and annotations
            need to undo-resize first !!!
            """
            current_frame_example = frames[0, -1].permute(1, 2, 0).numpy()
            current_frame_example *= 255
            cv2.imwrite('test.jpg', current_frame_example)
            cv2_image = cv2.imread('test.jpg')
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            nao_bbox_example = nao_bbox[0].numpy()
            cv2.rectangle(cv2_image, (nao_bbox_example[0], nao_bbox_example[1]),
                          (nao_bbox_example[2], nao_bbox_example[3]), (255, 0, 0), 3)
            #
            cv2.imshow(f'{current_frame_path[0][-10:-4]}', cv2_image)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('s'):
                save_path = os.path.join('/media/luohwu/T7/experiments/visualization', window_name.replace('/', '_'))
                cv2.imwrite(
                    filename=save_path,
                    img=cv2_image
                )
                cv2.destroyAllWindows()
            elif key == ord('q'):
                cv2.destroyAllWindows()
                break
            else:
                cv2.destroyAllWindows()
                continue


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
