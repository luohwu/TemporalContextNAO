import sys
sys.path.insert(0,'..')

from comet_ml import Experiment
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
from detectron2.modeling import build_model
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
experiment = Experiment(
    api_key="wU5pp8GwSDAcedNSr68JtvCpk",
    project_name="intent-net-with-temporal-context",
    workspace="thesisproject",
    auto_metric_logging=False
)


def compute_iou(bbox1,bbox2):
    area1=(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2=(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    inter_l=max(bbox1[0],bbox2[0])
    inter_r=min(bbox1[2],bbox2[2])
    inter_t=max(bbox1[1],bbox2[1])
    inter_b=min(bbox1[3],bbox2[3])
    inter_area = max((inter_r - inter_l),0) * max((inter_b - inter_t),0)
    return inter_area/(area1+area2-inter_area)

def none_maximum_suppression(ro_bboxes):
    length=len(ro_bboxes)
    if length<2:
        return ro_bboxes
    idx_to_remove=[]
    for i in range(length):
        for j in range(i+1,length):
            iou=compute_iou(ro_bboxes[i],ro_bboxes[j])
            # print(iou)
            if iou>0.6:
                idx_to_remove.append(i)
    result=[ro_bboxes[i] for i in range(length) if i not in idx_to_remove]
    # if len(idx_to_remove)>0:
    #     print(f'idx: {idx_to_remove}')
    #     print(f'bboxes: {ro_bboxes}')
    #     print(f'bboxes_nms: {result}')
    return result

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


    # df_items=df_items.rename(columns={'class':'category'})
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



def detect_relevant_objects(row,img_folder):
    img_path = os.path.join(img_folder, f"frame_{str(row['frame']).zfill(10)}.jpg")
    img=cv2.imread(img_path)
    # cv2.imshow(img_path[-30:],img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    outputs = predictor(img)
    pred_boxes = outputs['instances'].pred_boxes
    relevant_objs = []
    if len(pred_boxes) > 0:
        for box in pred_boxes:
            relevant_objs.append(box.int().cpu().numpy().tolist())
    # print(relevant_objs)
    return none_maximum_suppression(relevant_objs)

if __name__ == '__main__':
    if args.euler:
        import tarfile
        scratch_path = os.environ['TMPDIR']
        tar_path = f'/cluster/home/luohwu/{args.dataset_file}'
        assert os.path.exists(tar_path), f'file not exist: {tar_path}'
        print('extracting dataset from tar file')
        tar = tarfile.open(tar_path)
        tar.extractall(os.environ['TMPDIR'])
        tar.close()
        print('finished')
    data=make_sequence_dataset('all', args.dataset)
    print(type(data))
    print(data)
    classes = data['class'].unique()
    num_cls=len(classes)
    # classes = data['class'].unique()
    for d in ["train", "test"]:
        DatasetCatalog.register("nao_" + d, lambda d=d: get_nao_dicts(make_sequence_dataset(d,args.dataset)))
        MetadataCatalog.get("nao_" + d).set(thing_classes=classes)
    nao_train_metadata = MetadataCatalog.get("nao_train")
    print(nao_train_metadata)


    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Get the basic model configuration from the model zoo
    # Passing the Train and Validation sets
    cfg.DATASETS.TRAIN = ("nao_train")
    cfg.DATASETS.TEST = ()
    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = os.path.join('../output/', f"model_final_{args.dataset}.pth")   # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # Number of images per batch across all machines.
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_cls  # No. of classes = [HINDI, ENGLISH, OTHER]
    cfg.TEST.EVAL_PERIOD = 5  # No. of iterations after which the Validation Set is evaluated.
    # cfg.MODEL.DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.euler:
            cfg.MODEL.DEVICE='cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2 if args.dataset=='ADL' else 0.3  # set threshold for this model
    predictor = DefaultPredictor(cfg)

    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR,
    #                                  f"model_final_{args.dataset}.pth")  # Let training initialize from model zoo
    # predictor_resized = DefaultPredictor(cfg)

    par_video_id_list = sorted(id)
    for video_id in sorted(par_video_id_list):
        if args.dataset == 'EPIC':
            img_folder=os.path.join(args.data_path, frames_path, video_id[:3],video_id[3:])
            video_id = video_id[3:]
        else:
            img_folder=os.path.join(args.data_path, frames_path, video_id)
        anno_file_path = os.path.join(args.data_path, annos_path, f'nao_{video_id}.csv')
        if os.path.exists(anno_file_path):
            print(f'current video id: {video_id}')
            annotations = pd.read_csv(anno_file_path, converters={"nao_bbox": literal_eval})
            annotations['ro_bbox']=annotations.apply(detect_relevant_objects,args=[img_folder],axis=1)
            annotations.to_csv(anno_file_path, index=False)


    # dataset_dicts = get_nao_dicts(make_sequence_dataset('train',args.dataset))
    # for d in random.sample(dataset_dicts, 300):
    #     file_path=d["file_name"]
    #     img = cv2.imread(file_path)
    #     img_resized=cv2.resize(img,(224,224))
    #     outputs=predictor(img)
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=nao_train_metadata, scale=1.0)
    #     out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     # print(outputs["instances"])
    #     cv2.imshow('image',out.get_image()[:, :, ::-1])
    #     pred_boxes=outputs['instances'].pred_boxes
    #     print('=' * 50)
    #     relevant_objs = []
    #     if len(pred_boxes)>0:
    #         for box in pred_boxes:
    #             relevant_objs.append(box.numpy().tolist())
    #     print(relevant_objs)
    #
    #     # outputs_resized=predictor_resized(img_resized)
    #     # visualizer_resized = Visualizer(img_resized[:, :, ::-1], metadata=nao_train_metadata, scale=1.0)
    #     # out_resized = visualizer_resized.draw_instance_predictions(outputs_resized["instances"].to("cpu"))
    #     # # print(outputs["instances"])
    #     # cv2.imshow('image_resized',out_resized.get_image()[:, :, ::-1])
    #     # cv2.moveWindow('image_resized',700,300)
    #
    #     #
    #     # v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    #     # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     # cv2.imshow('image',out.get_image()[:, :, ::-1])
    #     key = cv2.waitKey(0) & 0xFF
    #     if key == ord('s'):
    #         print(d["file_name"])
    #         save_path = os.path.join('/media/luohwu/T7/experiments/faster_rcnn', d["file_name"][-27:].replace('/', '_'))
    #         print(save_path)
    #         cv2.imwrite(
    #             filename=save_path,
    #             img=out.get_image()[:, :, ::-1]
    #         )
    #         cv2.destroyAllWindows()
    #         plt.close()
    #     elif key == ord('q'):
    #         cv2.destroyAllWindows()
    #         plt.close()
    #         break
    #     else:
    #         cv2.destroyAllWindows()
    #         plt.close()
    #         continue