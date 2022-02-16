import os.path

from comet_ml import Experiment
from datetime import datetime
import matplotlib.pyplot as plt
import torch.cuda
from torch import optim
from torch.utils.data import DataLoader
from data.dataset import *
from opt import *
import tarfile
from torch import  nn
import pandas as pd
import cv2
import numpy as np
from tools.Schedulers import *
from data.dataset_ambiguity import NAODatasetBase
from models.IntentNetAmbiguity import *

experiment = Experiment(
    api_key="wU5pp8GwSDAcedNSr68JtvCpk",
    project_name="intentnetambiguity",
    workspace="thesisproject",
)
experiment.log_parameters(args.__dict__)
SEED = args.seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

exp_name = args.exp_name

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


multi_gpu = True if torch.cuda.device_count() > 1 else False
print(f'using {torch.cuda.device_count()} GPUs')
print('current graphics card is:')
os.system('lspci | grep VGA')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def my_collate(batch):
    frames_list=[]
    mask_list=[]
    frame_path_list=[]
    bbox_list=[]
    for item in batch:
        frames_list.append(item[0])
        bbox_list.append(item[1])
        frame_path_list.append(item[2])
        mask_list.append(item[3])
    return torch.stack(frames_list),bbox_list,(frame_path_list),torch.stack(mask_list)

def main():
    model=IntentNetBase()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'model size: {total_params}')
    if args.dataset=='ADL':
        model.load_state_dict(
            torch.load(f'{args.exp_path}/ADL/ambiguity/ckpts/model_epoch_150.pth', map_location='cpu')[
                'model_state_dict'], strict=False)
    else:
        model.load_state_dict(
            torch.load(f'{args.exp_path}/EPIC/ambiguity/ckpts/model_epoch_80.pth', map_location='cpu')[
                'model_state_dict'], strict=False)

    model = model.to(device)

    if args.original_split:
        train_dataset = NAODatasetBase(mode='train', dataset_name=args.dataset)
        test_dataset = NAODatasetBase(mode='test', dataset_name=args.dataset)
    else:
        all_data = NAODatasetBase(mode='all', dataset_name=args.dataset)
        train_size=int(0.8*len(all_data))
        test_size=len(all_data)-train_size
        train_dataset, test_dataset = torch.utils.data.random_split(all_data, [train_size, test_size],
                                                                    generator=torch.Generator().manual_seed(args.seed))


    # train_dataset, test_dataset = ini_datasets(dataset_name=args.dataset, original_split=args.original_split)



    print(f'train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}')
    test_dataloader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=True, num_workers=2,
                                pin_memory=True,
                                drop_last=True if torch.cuda.device_count() >= 4 else False,
                                 collate_fn=my_collate)

    if args.SGD:
        print('using SGD')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:

        print('using AdamW')
        optimizer = optim.AdamW(filter(lambda  p: p.requires_grad,model.parameters()),
                                lr=args.lr,
                                betas=(0.9, 0.99),
                                weight_decay=0
                                # ,weight_decay=args.weight_decay
                                )





    criterion = AttentionLoss()
    # criterion=nn.MSELoss()

    train_args['ckpt_path'] = os.path.join(train_args['exp_path'],args.dataset,
                                           exp_name, 'ckpts/')
    if not os.path.exists(train_args['ckpt_path']):
        os.mkdir(train_args['ckpt_path'])


    for epoch in range(1):
        print(f"==================epoch :{epoch}/{train_args['epochs']}===============================================")

        test_loss = eval(test_dataloader, model, criterion, epoch, illustration=True)




def eval(test_dataloader, model, criterion, epoch, illustration):
    model.eval()
    total_test_loss = 0
    top_1_all=0
    top_3_all=0
    len_dataset = len(test_dataloader.dataset)
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            frame,all_bboxes, img_path,mask = data
            frame=frame.to(device)
            mask=mask.to(device)

            output, decoder_feautre = model(frame)
            del frame
            if illustration:
                for i in range(mask.shape[0]):
                    top_1,top_3=compute_acc(output[i],all_bboxes[i])
                    top_1_all+=top_1
                    top_3_all+=top_3
                    mask=np.ones((256,456),dtype=np.uint8)*1
                    img_path_item=img_path[i]
                    original_image=cv2.imread(img_path_item)
                    original_image_bbox = cv2.imread(img_path_item)
                    original_image_bbox=cv2.rectangle(original_image_bbox,(all_bboxes[i][0][0],all_bboxes[i][0][1]),
                                                 (all_bboxes[i][0][2],all_bboxes[i][0][3]),(255,0,0),2)
                    mask[all_bboxes[i][0][1]:all_bboxes[i][0][3],all_bboxes[i][0][0]:all_bboxes[i][0][2]]=1
                    ro_bboxes=all_bboxes[i][1:]
                    for box in ro_bboxes:
                        original_image_bbox = cv2.rectangle(original_image_bbox, (box[0], box[1]),
                                                       (box[2], box[3]), (0, 255, 0), 2)
                        mask[box[1]:box[3],box[0]:box[2]]=1

                    output_item=output[i]*255
                    output_item=output_item.cpu().detach().numpy().astype(np.uint8)
                    output_item=output_item*mask
                    # output_item=heatmap_to_bbox(output_item)
                    output_item=cv2.applyColorMap(output_item,cv2.COLORMAP_JET)
                    masked_img=cv2.addWeighted(original_image,0.7,output_item,0.3,0)
                    saved_img=np.concatenate((original_image_bbox,masked_img),axis=1)
                    if top_1==1:
                        saved_path = os.path.join(f'/cluster/home/luohwu/experiments/{args.dataset}/ambiguity/top_1_correct/',
                                                  img_path_item[-25:].replace('/', '_'))
                    else:
                        saved_path = os.path.join(f'/cluster/home/luohwu/experiments/{args.dataset}/ambiguity/top_1_wrong/',
                                                  img_path_item[-25:].replace('/', '_'))
                    # print(saved_path)
                    cv2.imwrite(saved_path,saved_img)

        test_loss_avg = total_test_loss / len_dataset
    print(f'top-1: {top_1_all/len_dataset}, top-3: {top_3_all/len_dataset}')

    model.train()

    return test_loss_avg


def heatmap_to_bbox(heatmap):
    thresh = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(heatmap, (x, y), (x + w, y + h), (36, 255, 12), 2)

    return heatmap

def compute_iou(bbox1,bbox2):
    area1=(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2=(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    inter_l=max(bbox1[0],bbox2[0])
    inter_r=min(bbox1[2],bbox2[2])
    inter_t=max(bbox1[1],bbox2[1])
    inter_b=min(bbox1[3],bbox2[3])
    inter_area = max((inter_r - inter_l),0) * max((inter_b - inter_t),0)
    return inter_area/(area1+area2-inter_area)

def compute_acc(output,bboxes):
    num_bboxes=len(bboxes)-1
    if num_bboxes==0:
        return 0,0
    output=output.cpu().detach().numpy()
    attention_vector=np.zeros(num_bboxes)
    iou_vector=np.zeros(num_bboxes)
    gt=bboxes[0]
    for i,box in enumerate(bboxes[1:]):
        area=(box[2]-box[0]+1)*(box[3]-box[1]+1)
        attention=output[box[1]:box[3],box[0]:box[2]].sum()/(area)
        attention_vector[i]=attention
        iou_vector[i]=compute_iou(gt,box)
    desceending_index=np.argsort(attention_vector)[::-1]
    top_1=1 if iou_vector[desceending_index[0]]>0.5 else 0
    if num_bboxes>3:
        top_3= 1 if np.any(iou_vector[desceending_index][:3]>0.5) else 0
    else:
        top_3=1 if np.any(iou_vector>0.5) else 0
    return top_1,top_3


if __name__ == '__main__':

    main()

