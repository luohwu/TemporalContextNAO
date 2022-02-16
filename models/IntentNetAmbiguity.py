from comet_ml import Experiment

import sys
sys.path.insert(0,'..')

import cv2
import torch
import torchvision.transforms.transforms
from torch import nn
from torchvision import models
import torch.nn.functional as F
from opt import *

p_dropout=0.5 if args.dataset=='EPIC' else 0.3

print('current graphics card is:')
os.system('lspci | grep VGA')
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ConvBlock(nn.Module):
    def __init__(self,in_c,out_c,padding=1):
        super(ConvBlock, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3,padding=padding),
            # nn.BatchNorm2d(out_c),
            nn.Dropout2d(p_dropout),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,kernel_size=3,padding=padding),
            nn.BatchNorm2d(out_c),
        )

    def forward(self,x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self,in_channel=3,out_channels=[64,128,256,512,1024]):
        super(Encoder, self).__init__()
        channels=out_channels.copy()
        channels.insert(0,in_channel)
        self.conv_block_list=nn.ModuleList([ConvBlock(channels[i],channels[i+1]) for i in range(len(out_channels))])
        self.max_pool=nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        buffered_features=[]
        for block in self.conv_block_list:
            x=block(x)
            buffered_features.append(x)
            x=self.max_pool(x)
        return buffered_features

class Decoder(nn.Module):
    def __init__(self,in_channel=1024,out_channels=[512, 256, 128, 64]):
        super(Decoder, self).__init__()
        self.channels=out_channels.copy()
        self.channels.insert(0,in_channel)
        self.up_block_list=nn.ModuleList([nn.ConvTranspose2d(self.channels[i],self.channels[i+1],
                                             # kernel_size=(3,2) if i==0 else (2,2),stride=2)
                                             kernel_size=(2, 2) if i == 0 else (2, 2), stride=2)
                                              for i in range(len(out_channels))])
        self.conv_block_list=nn.ModuleList([
            ConvBlock(self.channels[i],self.channels[i+1]) for i in range(len(out_channels))
        ])
        self.dropout=nn.Dropout2d(p_dropout)

    def forward(self,x,buffered_features):
        for i in range(len(self.channels)-1):
            x=self.up_block_list[i](x)
            x=self.dropout(x)
            x=torch.cat([x,buffered_features[i]],dim=1)
            x=self.conv_block_list[i](x)
        return x

class IntentNetBase(nn.Module):
    def __init__(self):
        super(IntentNetBase, self).__init__()
        self.encoder=Encoder(in_channel=3,out_channels=[64,128,256,512])
        self.decoder=Decoder(in_channel=512,out_channels=[256,128,64])
        self.head=nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            nn.Sigmoid(),
        )
    def forward(self,x):
        encoder_features=self.encoder(x)
        feature_last_layer=encoder_features[-1]
        decoder_feature=self.decoder(feature_last_layer,encoder_features[:-1][::-1])
        output=self.head(decoder_feature)
        return output.squeeze(1),decoder_feature


class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.BCE=nn.BCELoss(reduction='none')

    def forward(self,outputs,targets,decoder_feature,bboxes):

        B,C,H,W=decoder_feature.shape
        weight_masks=torch.ones(B,H,W,requires_grad=False)*0.8
        weight_masks=weight_masks.to(device)
        with torch.no_grad():
            for i in range(B):
                nao_bbox = bboxes[i][0]
                feature_nao_bbox = F.adaptive_max_pool2d(
                    decoder_feature[i, :, nao_bbox[1]:nao_bbox[3], nao_bbox[0]:nao_bbox[2]], output_size=7)
                feature_nao_bbox = torch.flatten(feature_nao_bbox, 0).unsqueeze(0)
                ro_bbox = bboxes[i][1:]
                for box in ro_bbox:
                    # print(box)
                    feature_ro_bbox = F.adaptive_max_pool2d(decoder_feature[i, :, box[1]:box[3], box[0]:box[2]],
                                                            output_size=7)
                    feature_ro_bbox = torch.flatten(feature_ro_bbox, 0).unsqueeze(0)
                    cos_similarity = 0.8+torch.nn.functional.cosine_similarity(feature_ro_bbox, feature_nao_bbox)

                    # print(cos_similarity)
                    weight_masks[i,box[1]:box[3], box[0]:box[2]] = cos_similarity

        pixel_wise_loss=self.BCE(outputs,targets)*weight_masks
        return pixel_wise_loss.mean()

def main():
    from PIL import Image
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    model = IntentNetBase()
    model=model.to(device)


    transform = torchvision.transforms.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    if args.euler:
        img_file = '/cluster/home/luohwu/dataset/ADL/rgb_frames/P_04/frame_0000023400.jpg'
        mask_file = '/cluster/home/luohwu/dataset/ADL/rgb_frames/P_04/frame_0000023400.npy'

    else:
        img_file = '/media/luohwu/T7/dataset/ADL/rgb_frames/P_04/frame_0000023400.jpg'
        mask_file = '/media/luohwu/T7/dataset/ADL/rgb_frames/P_04/frame_0000023400.npy'

        img_file = '/home/luohwu/euler/dataset/ADL/rgb_frames/P_04/frame_0000023400.jpg'
        mask_file = '/home/luohwu/euler/dataset/ADL/rgb_frames/P_04/frame_0000023400.npy'

    all_bboxes=[
             [133, 172, 215, 220],
             [138, 175, 212, 216],
             # [252.09121704101562, 146.6275634765625, 298.5419616699219, 216.53521728515625],
             # [319.6934509277344, 139.83990478515625, 407.7671813964844, 217.42263793945312]
                ]
    # all_bboxes=torch.tensor(all_bboxes,dtype=torch.int)
    input = Image.open(img_file)
    # input=cv2.imread(img_file)
    input = transform(input)
    input=input.to(device)
    loss_fn = AttentionLoss()
    # loss_fn=nn.BCELoss()
    target_np = np.load(mask_file)
    # cv2.imshow('GT', target_np)
    target = torch.tensor(target_np).float()
    target=target.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=3e-4,
                                  betas=(0.9, 0.99))
    epoch = 1000

    #mimic one batch
    input=torch.stack([input,input]).to(device)
    all_bboxes=[all_bboxes,all_bboxes]
    target=torch.stack([target,target]).to(device)

    for i in range(epoch):
        output, decoder_feautre = model(input)
        # weight_numpy=weight.numpy()
        # cv2.imshow('weight',weight_numpy)
        # cv2.waitKey(0)
        loss = loss_fn(output, target,decoder_feautre,all_bboxes)
        print('='*50)
        print(f'epoch: {i}/1000, loss: {loss.item()}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output_numpy = output[0].cpu().detach().numpy()
        if args.euler:
            cv2.imwrite(f'/cluster/home/luohwu/experiments/draft/output_{i}.png', output_numpy * 255)
        else:
            cv2.imwrite(f'/media/luohwu/T7/experiments/draft/output_{i}.png', output_numpy * 255)
            cv2.imwrite(f'/home/luohwu/euler/experiments/draft/output_{i}.png', output_numpy * 255)
        # cv2.waitKey(0)
    # cv2.imshow('output', output_numpy)
    # cv2.waitKey(0)

    # print(input)
    # input=input.permute(1,2,0)
    # print(input)
    # plt.imshow(input)
    # cv2.imshow('test',input)
    # cv2.waitKey(0)

def compute_similarity(feature,bboxes):
    with torch.no_grad():
        nao_bbox=bboxes[0]
        feature_nao_bbox=F.adaptive_max_pool2d(feature[:,:,nao_bbox[1]:nao_bbox[3],nao_bbox[0]:nao_bbox[2]],output_size=7)
        feature_nao_bbox=torch.flatten(feature_nao_bbox,1)
        ro_bbox=bboxes[1:]
        weight=torch.ones(256,456,requires_grad=False)*0.4
        for box in ro_bbox:
            # print(box)
            feature_ro_bbox=F.adaptive_max_pool2d(feature[:,:,box[1]:box[3],box[0]:box[2]],output_size=7)
            feature_ro_bbox=torch.flatten(feature_ro_bbox,1)
            cos_similarity=torch.nn.functional.cosine_similarity(feature_ro_bbox,feature_nao_bbox)
            mse_loss=F.mse_loss(feature_ro_bbox,feature_nao_bbox,reduction='sum')

            print(cos_similarity)
            weight[box[1]:box[3],box[0]:box[2]]=cos_similarity
        # weight=weight=torch.ones(256,456)

    return weight



if __name__=='__main__':
    experiment = Experiment(
        api_key="wU5pp8GwSDAcedNSr68JtvCpk",
        project_name="intentnetambiguity",
        workspace="thesisproject",
    )
    main()
