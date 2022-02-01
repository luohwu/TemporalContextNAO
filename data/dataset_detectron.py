import time
from ast import literal_eval

import pandas as pd
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from opt import *

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import matplotlib.pyplot as plt
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os

def make_sequence_dataset(mode='train',dataset_name='ADL'):
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
                annos_subset = annos[['img_path', 'nao_bbox', 'class', 'previous_frames', 'frame']]
                df_items = df_items.append(annos_subset)



    # df_items = df_items.rename(columns={'nao_bbox_resized': 'nao_bbox'})
    print('finished')
    print('=============================================================')
    return df_items

def get_nao_dicts(data):
    classes=data['class'].unique()
    cls_to_index={classes[i]:i for i in range(len(classes))}
    print(data.shape)


    dataset_dicts = []
    for idx,item in data.iterrows():
        # print(item['class'])
        record = {}
        #
        filename = os.path.join(item['img_path'], f"frame_{str(item['frame']).zfill(10)}.jpg")
        height, width = cv2.imread(filename).shape[:2]
        #
        record["file_name"] = filename
        record["image_id"] = filename[-25:]
        record["height"] = height
        record["width"] = width
        #
        # annos = v["regions"]
        objs = []
        obj={
            "bbox":item['nao_bbox'],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id":cls_to_index[item['class']]
        }
        objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# def get_nao_dicts(mode):
#     data=make_sequence_dataset(mode=mode,dataset_name=args.dataset)
#
#     with open(json_file) as f:
#         imgs_anns = json.load(f)
#
#     dataset_dicts = []
#     for idx, v in enumerate(imgs_anns.values()):
#         record = {}
#
#         filename = os.path.join(img_dir, v["filename"])
#         height, width = cv2.imread(filename).shape[:2]
#
#         record["file_name"] = filename
#         record["image_id"] = idx
#         record["height"] = height
#         record["width"] = width
#
#         annos = v["regions"]
#         objs = []
#         for _, anno in annos.items():
#             assert not anno["region_attributes"]
#             anno = anno["shape_attributes"]
#             px = anno["all_points_x"]
#             py = anno["all_points_y"]
#             poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
#             poly = [p for x in poly for p in x]
#
#             obj = {
#                 "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
#                 "bbox_mode": BoxMode.XYXY_ABS,
#                 "segmentation": [poly],
#                 "category_id": 0,
#             }
#             objs.append(obj)
#         record["annotations"] = objs
#         dataset_dicts.append(record)
#     return dataset_dicts
#



if __name__ == '__main__':
    classes = make_sequence_dataset('all', 'ADL')['class'].unique()
    num_cls=len(classes)
    # classes = data['class'].unique()
    for d in ["train", "test"]:
        DatasetCatalog.register("nao_" + d, lambda d=d: get_nao_dicts(make_sequence_dataset(d,args.dataset)))
        MetadataCatalog.get("nao_" + d).set(thing_classes=classes)
    nao_train_metadata = MetadataCatalog.get("nao_train")
    print(nao_train_metadata)

    # dataset_dicts = get_nao_dicts(make_sequence_dataset('train',args.dataset))
    # for d in random.sample(dataset_dicts, 3):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=nao_train_metadata, scale=1.0)
    #     out = visualizer.draw_dataset_dict(d)
    #     cv2.imshow('image',out.get_image()[:, :, ::-1])
    #     # plt.imshow(out.get_image()[:, :, ::])
    #     # plt.show()
    #     cv2.waitKey(0)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Get the basic model configuration from the model zoo
    # Passing the Train and Validation sets
    cfg.DATASETS.TRAIN = ("nao_train")
    cfg.DATASETS.TEST = ()
    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LearningRate
    cfg.SOLVER.MAX_ITER = 1500  # No. of iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_cls  # No. of classes = [HINDI, ENGLISH, OTHER]
    cfg.TEST.EVAL_PERIOD = 5  # No. of iterations after which the Validation Set is evaluated.
    cfg.MODEL.DEVICE='cpu'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()