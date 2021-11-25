import torch
from torch import  nn
from torchvision import models
from mmcv import Config, DictAction
from mmaction.models import build_model
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from model.unet_resnet import UNetResNet18

class FuseBlock(nn.Module):
    def __init__(self,time_length):
        super(FuseBlock,self).__init__()
        self.time_length=time_length
        # [Batch_size,Time,C,H,W] -> [Batch_size,1,C,H,W]
        # compress information into only one time moment
        self.reduce_time=nn.Conv3d(int(time_length/2),1,kernel_size=1,stride=1)

        # channels of features extracted by ResNet20 is 2048
        # channels of temporal context is 1024
        self.cn=nn.Conv2d(2048,1024,kernel_size=1,stride=1)
        self.cn2=nn.Conv2d(1024,256,kernel_size=1,stride=1)

    def forward(self,context_features,visual_feature):
        # [B,C,T,H, W] -> [B,T,C,H,W]
        context_features=context_features.permute(0,2,1,3,4)
        compreseed_context_features=self.reduce_time(context_features)
        compreseed_context_features=compreseed_context_features.squeeze(1)
        visual_feature=self.cn(visual_feature)

        # C 1025 -> 256
        return self.cn2(compreseed_context_features+visual_feature)


class TemporalNaoNet(nn.Module):
    def __init__(self,time_length):
        super(TemporalNaoNet,self).__init__()
        self.temporal_length=time_length
        config = 'configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
        checkpoint = 'checkpoints/swin_base_patch244_window1677_sthv2.pth'
        cfg = Config.fromfile(config)
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location='cpu')
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)

        # input: [batch_size, channel, temporal_dim, height, width] e.g. (1, 3, 32, 224, 224)
        # output:  [batch_size, hidden_dim, temporal_dim/2, height/32, width/32] e.g.[1, 1024, 16, 7, 7]
        self.temporal_context_extractor=model.backbone
        # self.base_model=UNetResNet18()

        self.fuse_block=FuseBlock(time_length)

        self.MLP=nn.Sequential(nn.Linear(12544,4096),
                               nn.Linear(4096,4)
                               )

        self.flattern=nn.Flatten(1)


    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,previous_frames,current_frame):

        # context: [batch_size, hidden_dim, temporal_dim/2, height/32, width/32]
        context=self.temporal_context_extractor(previous_frames)
        # print(f'context shape: ',context.shape)

        # visual_feature: [batch_size, new_channels, height, width]
        # _, visual_feature = self.base_model(current_frame, with_output_feature_map=True)
        # print(f'visual feature shape: {visual_feature.shape}')

        # visual_feature: [batch_size, new_channels, height, width]
        # print(f'current frame shape: {current_frame.shape}')
        visual_feature = self.visual_feature(current_frame)
        # print(f'visual feature shape: {visual_feature.shape}')

        fused_feature=self.fuse_block(context,visual_feature)
        # print(f'fused features shape',fused_feature.shape)


        return self.MLP(self.flattern(fused_feature))


if __name__=='__main__':
    # resnet = models.resnet50(pretrained=True)
    # modules = list(resnet.children())[:-2]
    # resnet=nn.Sequential(*modules)
    # img=torch.rand(1,3,224,224)
    # output=resnet(img)
    # print(output.shape)


    model=TemporalNaoNet(time_length=10)
    total_params = sum(p.numel() for p in model.parameters())-sum(p.numel() for p in model.temporal_context_extractor.parameters())
    print(f'total # of parameters: {total_params}')  # 参数总数: 19341058
    previous_frames=torch.rand(4, 3, 10, 224, 224)
    current_frame=torch.rand(4,3,224,224)
    output=model(previous_frames,current_frame)
    print(output.shape)
