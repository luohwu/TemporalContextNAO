import torch
from torch import  nn
from torchvision import models
from mmcv import Config
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
from torch.nn import  init
from models.unet_resnet_backup import UNetResNet18

from swin_extractor import SwinTransformer



class IntentNetBase(nn.Module):
    def __init__(self):
        super(IntentNetBase, self).__init__()
        resnet=models.resnet18(pretrained=True)
        # resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())
        self.visual_feature = nn.Sequential(*modules[:-3],
                                            modules[-3][0])
        self.head=nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
            # nn.AdaptiveAvgPool1d(512),

            # nn.Linear(2048,512),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            # nn.Linear(1024,512),
            # nn.Dropout(0.5),
            # nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(128,4),
            nn.Sigmoid()
            # nn.Linear(512,256),
            # nn.Dropout(0.5),
            # nn.LeakyReLU(),
            # nn.Linear(256,4),
            # nn.Sigmoid()
        )



    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,frames):
        # current_frame=frames[:,-1]
        current_frame = frames


        visual_feature = self.visual_feature(current_frame)
        # print(f'visual feature shape: {visual_feature.shape}')
        head=self.head(visual_feature)
        # print(f'head shape: {head.shape}')
        return head*torch.tensor([456,256,456,256]).to(device)

class IntentNetBaseGlobal(nn.Module):
    def __init__(self):
        super(IntentNetBaseGlobal, self).__init__()
        resnet=models.resnet18(pretrained=True)
        # resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())
        # self.visual_feature = nn.Sequential(*modules[:-3],
        #                                     modules[-3][0])
        # self.visual_feature = nn.Sequential(*modules[:4],
        #                                     modules[4],
        #                                     nn.Dropout2d(0.7),
        #                                     modules[5],
        #                                     nn.Dropout2d(0.5),
        #                                     modules[6],
        #                                     nn.Dropout2d(0.5),
        #                                     modules[7][0],
        #                                     nn.Dropout(0.5))
        self.visual_feature = nn.Sequential(*modules[:4],
                                            modules[4][0],
                                            nn.Dropout2d(0.2),
                                            modules[4][1],
                                            nn.Dropout2d(0.2),
                                            modules[5][0])
        self.head1=nn.Sequential(
            # nn.Conv2d(512,512,kernel_size=(3,3),stride=1,padding=1),
            # nn.Dropout2d(0.7),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(),
            # nn.Conv2d(512,512,kernel_size=(3,3),stride=1,padding=1),
            # nn.BatchNorm2d(512),
            # nn.LeakyReLU(),
        )
        self.head2=nn.Sequential(
            nn.Conv2d(128, 64, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(64,4,1,1),
            # nn.BatchNorm2d(128),
            # nn.LeakyReLU(),
            # nn.Conv2d(128,4,1,1),
            # nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
            nn.Sigmoid()
        )



    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,frames):
        # current_frame=frames[:,-1]
        current_frame = frames


        visual_feature = self.visual_feature(current_frame)
        # print(f'visual feature shape: {visual_feature.shape}')
        head=self.head2(visual_feature+self.head1(visual_feature))
        # print(f'head shape: {head.shape}')
        return head*torch.tensor([456,256,456,256]).to(device)



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
        self.cn2=nn.Conv2d(512,1,kernel_size=1,stride=1)

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



config = 'configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
checkpoint = 'checkpoints/swin_base_patch244_window1677_sthv2.pth'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class IntentNet(nn.Module):
    def __init__(self):
        super(IntentNet,self).__init__()
        # resnet=models.resnet18(pretrained=True)
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.visual_neck=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1)
        )

        resnet3d=models.video.mc3_18(pretrained=True)
        modules=list(resnet3d.children())[:-1]
        self.temporal_context=nn.Sequential(*modules)
        self.temporal_context_neck=nn.Sequential(
            nn.Flatten(1)
        )

        self.head=nn.Sequential(
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,4),
            nn.Sigmoid()
        )


    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,previous_frames,current_frame):

        temporal_context=self.temporal_context(previous_frames)
        temporal_context=self.temporal_context_neck(temporal_context)
        # print(temporal_context.shape)
        visual_feature = self.visual_feature(current_frame)
        visual_feature=self.visual_neck(visual_feature)
        # print(visual_feature.shape)

        whole_feature=torch.cat((temporal_context,visual_feature),dim=-1)
        # print(whole_feature.shape)

        # return self.head(whole_feature)

        return self.head(whole_feature) * torch.tensor([456, 256, 456, 256]).to(device)


class IntentNetFuse(nn.Module):
    def __init__(self):
        super(IntentNetFuse,self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.visual_neck=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
        )
        self.visual_neck.apply(self.init_weights)

        resnet3d=models.video.mc3_18(pretrained=True)
        modules=list(resnet3d.children())[:-1]
        self.temporal_context=nn.Sequential(*modules)
        self.temporal_context_neck=nn.Sequential(
            nn.Flatten(1)
        )
        self.temporal_context_neck.apply(self.init_weights)


        self.fuse_block=nn.Sequential(
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512)
        )
        self.fuse_block.apply(self.init_weights)
        self.head=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Sigmoid()
        )
        self.head.apply(self.init_weights)



    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,previous_frames,current_frame):

        temporal_context=self.temporal_context(previous_frames)
        temporal_context=self.temporal_context_neck(temporal_context)
        # print(temporal_context.shape)
        visual_feature = self.visual_feature(current_frame)
        visual_feature=self.visual_neck(visual_feature)
        # print(visual_feature.shape)

        fused_feature=self.fuse_block(visual_feature+temporal_context)
        # print(whole_feature.shape)

        # return self.head(fused_feature+visual_feature)

        return self.head(fused_feature+visual_feature) * torch.tensor([456, 256, 456, 256]).cuda()

    def init_weights(self,m):
        pass
        # if isinstance(m,nn.Linear):
            # torch.nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.01)



class IntentNetIC(nn.Module):
    def __init__(self):
        super(IntentNetIC,self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.visual_neck=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
        )

        resnet3d=models.video.mc3_18(pretrained=True)
        modules=list(resnet3d.children())[:-1]
        self.temporal_context=nn.Sequential(*modules)
        # print(self.temporal_context)
        self.temporal_context_neck=nn.Sequential(
            nn.Flatten(1)
        )

        self.head=nn.Sequential(
            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,4),
            nn.Sigmoid()
        )


    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,previous_frames,current_frame):

        temporal_context=self.temporal_context(previous_frames)
        temporal_context=self.temporal_context_neck(temporal_context)
        # print(temporal_context.shape)
        visual_feature = self.visual_feature(current_frame)
        visual_feature=self.visual_neck(visual_feature)
        # print(visual_feature.shape)

        whole_feature=torch.cat((temporal_context,visual_feature),dim=-1)
        # print(whole_feature.shape)

        # return self.head(whole_feature)
        #
        return self.head(whole_feature) * torch.tensor([456, 256, 456, 256]).to(device)







class IntentNetFuseAttentionVector(nn.Module):
    def __init__(self):
        super(IntentNetFuseAttentionVector, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.visual_neck=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
        )
        self.visual_neck.apply(self.init_weights)



        self.fuse_block=nn.Sequential(
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512)
        )
        self.fuse_block.apply(self.init_weights)
        self.head=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Sigmoid()
        )
        self.head.apply(self.init_weights)
        self.attention_vector=torch.nn.Parameter(torch.rand(10))


    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,frames):
        previous_frames=frames[:,:-1]
        current_frame=frames[:,-1].squeeze(1)
        # previous_frames=previous_frames.transpose(1,2)
        B,T,C,H,W=previous_frames.shape

        previous_frames=previous_frames.reshape(-1,C,H,W) # [B*T, C, H, W]
        # print(f'new shape of previous frames{previous_frames.shape}')

        # apply the base models to extract visual features for each frame
        frame_wise_feature=self.visual_neck(self.visual_feature(previous_frames))

        # shape to [B, T, length_of_feature]
        frame_wise_feature=frame_wise_feature.view(B,T,-1)
        # print(f'frame-wise feature shape: {frame_wise_feature.shape}')

        # apply the attention operation
        # temporal context = linear combination of frame-wise features
        temporal_context=torch.matmul(torch.nn.functional.softmax(self.attention_vector,dim=0), frame_wise_feature)

        visual_feature = self.visual_feature(current_frame)
        visual_feature=self.visual_neck(visual_feature)
        # print('shape of visual feature neck',visual_feature.shape)

        fused_feature=self.fuse_block(visual_feature+temporal_context)
        # print(fused_feature.shape)

        # return self.head(fused_feature+visual_feature)

        return self.head(fused_feature+visual_feature) * torch.tensor([456, 256, 456, 256])
    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class IntentNetFuseAttentionVectorTest(nn.Module):
    def __init__(self):
        super(IntentNetFuseAttentionVectorTest, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.visual_neck=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
        )
        self.visual_neck.apply(self.init_weights)



        self.fuse_block=nn.Sequential(
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512)
        )
        self.fuse_block.apply(self.init_weights)
        self.head=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Sigmoid()
        )
        self.head.apply(self.init_weights)
        self.attention_vector=torch.nn.Parameter(torch.rand(10))


    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,previous_frames,current_frame):

        visual_feature = self.visual_feature(current_frame)
        visual_feature=self.visual_neck(visual_feature)
        # print('shape of visual feature neck',visual_feature.shape)

        frame_wise_feature=torch.stack([visual_feature.clone() for p in range(10)])
        frame_wise_feature=frame_wise_feature.transpose(0,1)
        # print(f'frame-wise feature shape: {frame_wise_feature.shape}')

        # apply the attention operation
        # temporal context = linear combination of frame-wise features
        temporal_context=torch.matmul(torch.nn.functional.softmax(self.attention_vector,dim=0), frame_wise_feature)

        fused_feature=self.fuse_block(visual_feature+temporal_context)
        # print(fused_feature.shape)

        # return self.head(fused_feature+visual_feature)

        return self.head(fused_feature+visual_feature) * torch.tensor([456, 256, 456, 256]).cuda()

    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class IntentNetFuseAttentionVectorSW(nn.Module):
    def __init__(self):
        super(IntentNetFuseAttentionVectorSW, self).__init__()
        self.visual_feature = SwinTransformer()



        self.fuse_block=nn.Sequential(
            nn.Linear(49,49),
            nn.BatchNorm1d(49),
            nn.ReLU(),
        )
        self.fuse_block.apply(self.init_weights)
        self.head=nn.Sequential(
            nn.Linear(49,4),
            nn.Sigmoid()
        )
        self.head.apply(self.init_weights)
        self.attention_vector=torch.nn.Parameter(torch.rand(10))


    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,previous_frames,current_frame):
        previous_frames=previous_frames.transpose(1,2)
        B,T,C,H,W=previous_frames.shape

        previous_frames=previous_frames.reshape(-1,C,H,W) # [B*T, C, H, W]
        # print(f'new shape of previous frames{previous_frames.shape}')

        # apply the base models to extract visual features for each frame
        frame_wise_feature=(self.visual_feature(previous_frames))

        # shape to [B, T, length_of_feature]
        frame_wise_feature=frame_wise_feature.view(B,T,-1)
        # print(f'frame-wise feature shape: {frame_wise_feature.shape}')

        # apply the attention operation
        # temporal context = linear combination of frame-wise features
        temporal_context=torch.matmul(torch.nn.functional.softmax(self.attention_vector,dim=0), frame_wise_feature)

        visual_feature = self.visual_feature(current_frame)
        # print('shape of visual feature neck',visual_feature.shape)

        fused_feature=self.fuse_block(visual_feature+temporal_context)
        # print(fused_feature.shape)

        # return self.head(fused_feature+visual_feature)

        return self.head(fused_feature+visual_feature) * torch.tensor([456, 256, 456, 256]).cuda()

    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class IntentNetFuseAttentionMatrix(nn.Module):
    def __init__(self):
        super(IntentNetFuseAttentionMatrix, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.visual_neck=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
        )
        self.visual_neck.apply(self.init_weights)



        self.fuse_block=nn.Sequential(
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512)
        )
        self.fuse_block.apply(self.init_weights)
        self.head=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Sigmoid()
        )
        self.head.apply(self.init_weights)
        self.attention_table=torch.nn.Parameter(torch.rand(10,512))


    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,previous_frames,current_frame):
        previous_frames=previous_frames.transpose(1,2)
        # print(f'shape of previous_frames: {previous_frames.shape}')
        B,T,C,H,W=previous_frames.shape

        previous_frames=previous_frames.reshape(-1,C,H,W) # [B*T, C, H, W]
        # print(f'new shape of previous frames{previous_frames.shape}')

        # apply the base models to extract visual features for each frame
        frame_wise_feature=self.visual_neck(self.visual_feature(previous_frames))
        # print(f'shape of frame_wise_feature: {frame_wise_feature.shape}')

        # shape to [B, T, length_of_feature]
        frame_wise_feature=frame_wise_feature.view(B,T,-1)
        # apply the attention operation
        # temporal context = linear combination of frame-wise features

        # [B,T,C]
        temporal_context=torch.nn.functional.softmax(self.attention_table,dim=0)* frame_wise_feature

        # [B,C]
        temporal_context=temporal_context.sum(dim=1)

        visual_feature = self.visual_feature(current_frame)
        visual_feature=self.visual_neck(visual_feature)
        # print('shape of visual feature neck',visual_feature.shape)

        fused_feature=self.fuse_block(visual_feature+temporal_context)
        # print(fused_feature.shape)

        # return self.head(fused_feature+visual_feature)

        return self.head(fused_feature+visual_feature) * torch.tensor([456, 256, 456, 256])

    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class IntentNetDataDrivenAttention(nn.Module):
    def __init__(self):
        super(IntentNetDataDrivenAttention, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.visual_neck=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
        )
        self.visual_neck.apply(self.init_weights)



        self.fuse_block=nn.Sequential(
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512)
        )
        self.fuse_block.apply(self.init_weights)
        self.head=nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Sigmoid()
        )
        self.head.apply(self.init_weights)
        self.attention_vector=torch.nn.Parameter(torch.rand(10))
        self.attention_layer=FrameAttentionBlock(feature_dim=512)
        self.attention_layer.apply(self.init_weights)


    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,previous_frames,current_frame):
        previous_frames=previous_frames.transpose(1,2)
        B,T,C,H,W=previous_frames.shape

        previous_frames=previous_frames.reshape(-1,C,H,W) # [B*T, C, H, W]
        # print(f'new shape of previous frames{previous_frames.shape}')

        # apply the base models to extract visual features for each frame
        frame_wise_feature=self.visual_neck(self.visual_feature(previous_frames))

        # shape to [B, T, length_of_feature]
        frame_wise_feature=frame_wise_feature.view(B,T,-1)


        visual_feature = self.visual_feature(current_frame)
        visual_feature=self.visual_neck(visual_feature)

        temporal_context=self.attention_layer(visual_feature,frame_wise_feature)
        # fused_feature = self.fuse_block(visual_feature + temporal_context)
        # print(fused_feature.shape)

        # return self.head(fused_feature+visual_feature)

        # return self.head(fused_feature + visual_feature) * torch.tensor([456, 256, 456, 256]).cuda()
        return self.head(temporal_context) * torch.tensor([456, 256, 456, 256]).cuda()

    # print('shape of visual feature neck',visual_feature.shape)


    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class FrameAttentionBlock(nn.Module):
    def __init__(self,feature_dim=512):
        super(FrameAttentionBlock,self).__init__()
        self.q=nn.Linear(feature_dim,feature_dim)
        self.kv=nn.Linear(feature_dim,feature_dim*2)
        self.scale=feature_dim**(-0.5)
        self.norm=nn.BatchNorm1d(feature_dim)

    def forward(self,current_feature,context_features):
        B,T,d=context_features.shape

        #[B d] -> [B d]
        q=self.q(current_feature)
        # q=self.norm(q)

        # print(f'shape of q: {q.shape}')

        #[B T d] -> [B T 2d] -> [B T 2 d]-> [2 B T d]
        kv=self.kv(context_features).reshape(B,T,2,-1).permute(2,0,1,3)
        k=kv[0]
        # k=self.norm(k)
        v=kv[1]
        # v=self.norm(v)
        # print(f'shape of k: {k.shape}')
        # print(f'shape of v: {v.shape}')

        # v.shape=[B T d]
        # q.shape=[B d]
        # q.unsqueeze(2).shape= [B d 1]
        # torch.matmul(v,q.unsqueeze(2)).shape= [B T 1]
        attention=(torch.matmul(v,q.unsqueeze(2))*self.scale)

        # attention.shape= [B 1 T]
        attention=torch.nn.functional.softmax(attention,dim=1).transpose(1,2)
        # print(f'shape of attention {attention.shape}')

        summary_context=torch.matmul(attention,v)
        # print(f'shape of summary_context: {summary_context.shape}')
        return self.norm(summary_context.squeeze(1))









class IntentNetFullAttention(nn.Module):
    def __init__(self):
        super(IntentNetFullAttention, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.visual_neck = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
        )
        # self.visual_neck.apply(self.init_weights)

        self.fuse_block = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        # self.fuse_block.apply(self.init_weights)
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )
        # self.head.apply(self.init_weights)
        self.attention_vector = torch.nn.Parameter(torch.rand(11))

    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self, previous_frames, current_frame):
        previous_frames = previous_frames.transpose(1, 2)
        B, T, C, H, W = previous_frames.shape
        current_frame = current_frame.unsqueeze(1)

        # [B,T+1,C,H,W]
        all_frames = torch.cat((previous_frames, current_frame), dim=1)
        all_frames = all_frames.reshape(-1, C, H, W)  # [B*(T+1), C, H, W]


        # apply the base models to extract visual features for each frame
        frame_wise_feature = self.visual_neck(self.visual_feature(all_frames))

        # shape to [B, T, length_of_feature]
        frame_wise_feature = frame_wise_feature.view(B, T + 1, -1)
        # print(f'frame-wise feature shape: {frame_wise_feature.shape}')

        # # apply the attention operation
        # # temporal context = linear combination of frame-wise features
        fused_feature = torch.matmul(torch.nn.functional.softmax(self.attention_vector, dim=0), frame_wise_feature)

        # fused_feature=self.fuse_block(visual_feature+temporal_context)
        # # print(fused_feature.shape)

        # # return self.head(fused_feature+visual_feature)

        return self.head(fused_feature) * torch.tensor([456, 256, 456, 256]).cuda()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

class IntentNetFuseHeatmap(nn.Module):
    def __init__(self):
        super(IntentNetFuseHeatmap,self).__init__()
        self.model=UNetResNet18()


    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,previous_frames,current_frame):
        return self.model(current_frame)

    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class IntentNetSwin(nn.Module):
    def __init__(self,time_length):
        super(IntentNetSwin, self).__init__()
        self.temporal_length=time_length
        cfg = Config.fromfile(config)
        model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, checkpoint, map_location='cpu')
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)

        # input: [batch_size, channel, temporal_dim, height, width] e.g. (1, 3, 32, 224, 224)
        # output:  [batch_size, hidden_dim, temporal_dim/2, height/32, width/32] e.g.[1, 1024, 16, 7, 7]
        self.temporal_context_extractor=model.backbone
        # self.base_model=UNetResNet18()

        self.fuse_block=FuseBlock(time_length)

        self.MLP=nn.Sequential(
            nn.Linear(49,4),
            nn.Sigmoid()
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


        return self.MLP(self.flattern(fused_feature))* torch.tensor([456, 256, 456, 256]).to(device)



if __name__=='__main__':

    config = '../configs/recognition/swin/swin_base_patch244_window1677_sthv2.py'
    checkpoint = '../checkpoints/swin_base_patch244_window1677_sthv2.pth'

    # models=IntentNetIC()
    # models=IntentNetFuseAttentionMatrix()
    model=IntentNetBaseGlobal()
    print(model.visual_feature)
    print('='*50)
    print(model.visual_feature[0])
    # models = IntentNetSwin(time_length=10)
    # num_params_temporal_context_extractor=sum(p.numel() for p in models.temporal_context_extractor.parameters())
    total_params = sum(p.numel() for p in model.parameters())-sum(p.numel() for p in model.visual_feature.parameters())
    print(f'alive size: {total_params}')
    # 99K and 11K, so the temporal feature extractor has more then 80K parameters
    # print(f'total # of parameters: {total_params}, total # of parameters without temporal feature: {total_params-num_params_temporal_context_extractor}')
    frames=torch.rand(2,3,224,224)
    output=model(frames)
    # outputs_history=models(previous_frames)
    # print(outputs_history.shape)
    print(output.shape)
