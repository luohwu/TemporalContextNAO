import cv2
import torch
import torchvision.transforms.transforms
from torch import nn
from torchvision import models



class ConvBlock(nn.Module):
    def __init__(self,in_c,out_c,padding=1):
        super(ConvBlock, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size=3,padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,kernel_size=3,padding=padding)
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

    def forward(self,x,buffered_features):
        for i in range(len(self.channels)-1):
            x=self.up_block_list[i](x)
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
            # nn.Sigmoid(),
        )
    def forward(self,x):
        encoder_features=self.encoder(x)
        feature_last_layer=encoder_features[-1]
        decoder_feature=self.decoder(feature_last_layer,encoder_features[:-1][::-1])
        output=self.head(decoder_feature)
        return output.squeeze(1)


class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
        self.BCE=nn.BCEWithLogitsLoss(reduction='none')

    def forward(self,outputs,targets):
        pixel_wise_loss=self.BCE(outputs,targets)
        return pixel_wise_loss.mean()




if __name__=='__main__':
    from PIL import Image
    import torchvision
    import matplotlib.pyplot as plt
    import numpy as np
    model=IntentNetBase()

    transform=torchvision.transforms.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    img_file='/media/luohwu/T7/dataset/ADL/rgb_frames/P_01/frame_0000000138.jpg'
    mask_file='/media/luohwu/T7/dataset/ADL/rgb_frames/P_01/frame_0000000138.npy'
    input=Image.open(img_file)
    # input=cv2.imread(img_file)
    input=transform(input).unsqueeze(0)
    loss_fn=AttentionLoss()
    # loss_fn=nn.BCELoss()
    target_np=np.load(mask_file)
    cv2.imshow('GT',target_np)
    target=torch.tensor(target_np).unsqueeze(0).float()
    optimizer=torch.optim.AdamW(model.parameters(),
                                lr=3e-4,
                                betas=(0.9,0.99))
    epoch=100
    for i in range(epoch):
        output = model(input)
        loss=loss_fn(output,target)
        print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output_numpy=output[0].detach().numpy()
        cv2.imwrite(f'/media/luohwu/T7/experiments/draft/output_{i}.png', output_numpy*255)
        # cv2.waitKey(0)
    # cv2.imshow('output', output_numpy)
    # cv2.waitKey(0)



    # print(input)
    # input=input.permute(1,2,0)
    # print(input)
    # plt.imshow(input)
    # cv2.imshow('test',input)
    # cv2.waitKey(0)