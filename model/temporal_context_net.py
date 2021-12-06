import torch
from torch import  nn
from torchvision import models
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class FuseBlock(nn.Module):
    def __init__(self,time_length):
        super(FuseBlock,self).__init__()
        self.time_length=time_length
        # [Batch_size,Time,C,H,W] -> [Batch_size,1,C,H,W]
        # compress information into only one time moment
        self.reduce_time=nn.Conv3d(int(time_length/2),1,kernel_size=1,stride=1)

        # channels of features extracted by ResNet20 is 2048
        # channels of temporal context is 1024
        self.cn=nn.Conv2d(1024,1,kernel_size=1,stride=1)
        self.cn2=nn.Conv2d(2048,1,kernel_size=1,stride=1)

    def forward(self,context_features,visual_feature):
        # [B,C,T,H, W] -> [B,T,C,H,W]
        context_features=context_features.permute(0,2,1,3,4)
        compreseed_context_features=self.reduce_time(context_features)
        compreseed_context_features=compreseed_context_features.squeeze(1)
        # visual_feature=self.cn(visual_feature)
        # print(compreseed_context_features.shape)
        # print(visual_feature.shape)

        # C 1025 -> 256
        return self.cn(compreseed_context_features)+self.cn2(visual_feature)
        # return self.cn2(compreseed_context_features+visual_feature)



class TemporalNaoNet(nn.Module):
    def __init__(self):
        super(TemporalNaoNet,self).__init__()
        # resnet=models.resnet18(pretrained=True)
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.head=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
            # resnet18 output 512 channels
            # resnet50 output 2048 channels
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024, 4),
        )



    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,current_frame):


        visual_feature = self.visual_feature(current_frame)
        # print(f'visual feature shape: {visual_feature.shape}')
        head=self.head(visual_feature)
        # print(f'head shape: {head.shape}')
        return torch.clamp(head,min=0,max=1)





if __name__=='__main__':

    model=TemporalNaoNet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'# of parameters: {total_params}')
    current_frame=torch.rand(2,3,224,224)
    targets=torch.rand((2,4))
    loss_fn=torch.nn.MSELoss()
    print(f'target: {targets}')
    output=model(current_frame)
    loss=loss_fn(output,targets)
    loss.backward()

    print(output)
    print(output.shape)
