import torch
from torch import  nn
from torchvision import models
import math
from swin_extractor import SwinTransformer
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



device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)  # embedding 与positional相加
        return self.dropout(x)




class IntentNetDataAttention(nn.Module):
    def __init__(self):
        super(IntentNetDataAttention, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.visual_neck=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
        )

        resnet2 = models.resnet18(pretrained=True)
        modules2 = list(resnet2.children())[:-2]
        self.visual_feature2 = nn.Sequential(*modules2)
        self.visual_neck2=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
        )


        self.visual_neck.apply(self.init_weights)



        self.fuse_block=nn.Sequential(
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,512),
            # nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.fuse_block.apply(self.init_weights)
        self.head=nn.Sequential(
            nn.Linear(512,256),

            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Sigmoid()
        )
        self.head.apply(self.init_weights)
        self.attention=Attention(dim=512,num_heads=8,p_drop_att=0.0,p_drop_ffn=0.0,depth=4,time_length=11,hidden_times=4)
        self.attention.apply(self.init_weights)

    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,frames):
        previous_frames=frames[:,:-1]
        current_frame=frames[:,-1].squeeze(1)
        B,T,C,H,W=previous_frames.shape

        previous_frames=previous_frames.reshape(-1,C,H,W) # [B*T, C, H, W]
        # print(f'new shape of previous frames{previous_frames.shape}')

        # apply the base models to extract visual features for each frame
        frame_wise_feature=self.visual_neck2(self.visual_feature2(previous_frames))

        # shape to [B, T, length_of_feature]
        frame_wise_feature=frame_wise_feature.view(B,T,-1)

        # print(f'shape of frame_wise_feature  {frame_wise_feature.shape}')


        visual_feature = self.visual_feature(current_frame)
        visual_feature=self.visual_neck(visual_feature)
        # print(f'shape of visual feature {visual_feature.shape}')

        # temporal_context=self.attention(torch.cat((frame_wise_feature,visual_feature.unsqueeze(1)),dim=1))
        temporal_context,atten=self.attention(torch.cat((frame_wise_feature,visual_feature.unsqueeze(1)),dim=1))
        fused_feature = self.fuse_block(visual_feature + temporal_context)
        # print(fused_feature.shape)

        # return self.head(fused_feature+visual_feature)

        # return self.head(temporal_context + visual_feature) * torch.tensor([456, 256, 456, 256]).cuda()
        return self.head(fused_feature + visual_feature) * torch.tensor([456, 256, 456, 256]).to(device),atten
        # return self.head(temporal_context) * torch.tensor([456, 256, 456, 256])

    # print('shape of visual feature neck',visual_feature.shape)


    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)



# class IntentNetDataAttentionSW(nn.Module):
#     def __init__(self):
#         super(IntentNetDataAttentionSW, self).__init__()
#         self.visual_feature = SwinTransformer()
#
#         self.visual_feature2 = SwinTransformer()
#
#
#
#
#
#         self.fuse_block=nn.Sequential(
#             nn.Linear(512,512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512,512)
#         )
#         self.fuse_block.apply(self.init_weights)
#         self.head=nn.Sequential(
#             nn.Linear(512,256),
#
#             # nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256,128),
#             # nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Linear(128,4),
#             nn.Sigmoid()
#         )
#         self.head.apply(self.init_weights)
#         self.attention=Attention(dim=512,num_heads=8,p_drop_att=0.0,p_drop_ffn=0.0,depth=6,time_length=11,hidden_times=3)
#         self.attention.apply(self.init_weights)
#
#     # previous_frames: [batch_size, channel, temporal_dim, height, width]
#     # current frame: [batch_sze, channel, height, width]
#     def forward(self,previous_frames,current_frame):
#         previous_frames=previous_frames.transpose(1,2)
#         B,T,C,H,W=previous_frames.shape
#
#         previous_frames=previous_frames.reshape(-1,C,H,W) # [B*T, C, H, W]
#         # print(f'new shape of previous frames{previous_frames.shape}')
#
#         # apply the base models to extract visual features for each frame
#         frame_wise_feature=(self.visual_feature2(previous_frames))
#
#         # shape to [B, T, length_of_feature]
#         frame_wise_feature=frame_wise_feature.view(B,T,-1)
#
#         # print(f'shape of frame_wise_feature  {frame_wise_feature.shape}')
#
#
#         visual_feature = self.visual_feature(current_frame)
#         # print(f'shape of visual feature {visual_feature.shape}')
#
#         temporal_context=self.attention(torch.cat((frame_wise_feature,visual_feature.unsqueeze(1)),dim=1))
#         fused_feature = self.fuse_block(visual_feature + temporal_context)
#         # print(fused_feature.shape)
#
#         # return self.head(fused_feature+visual_feature)
#
#         # return self.head(temporal_context + visual_feature) * torch.tensor([456, 256, 456, 256]).cuda()
#         return self.head(fused_feature + visual_feature) * torch.tensor([456, 256, 456, 256]).cuda()
#         # return self.head(temporal_context) * torch.tensor([456, 256, 456, 256])
#
#     # print('shape of visual feature neck',visual_feature.shape)
#
#
#     def init_weights(self,m):
#         if isinstance(m,nn.Linear):
#             torch.nn.init.xavier_uniform_(m.weight)
#             m.bias.data.fill_(0.01)


class IntentNetDataAttentionCat(nn.Module):
    def __init__(self):
        super(IntentNetDataAttentionCat, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.visual_feature = nn.Sequential(*modules)
        self.visual_neck=nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1),
        )

        resnet2 = models.resnet18(pretrained=True)
        modules2 = list(resnet2.children())[:-2]
        self.visual_feature2 = nn.Sequential(*modules2)
        self.visual_neck2=nn.Sequential(
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
            nn.Linear(1024,512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,256),

            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,4),
            nn.Sigmoid()
        )
        self.head.apply(self.init_weights)
        self.attention=Attention(dim=512,num_heads=8,p_drop_att=0.0,p_drop_ffn=0.0,depth=6,time_length=11,hidden_times=3)
        self.attention.apply(self.init_weights)

    # previous_frames: [batch_size, channel, temporal_dim, height, width]
    # current frame: [batch_sze, channel, height, width]
    def forward(self,previous_frames,current_frame):
        previous_frames=previous_frames.transpose(1,2)
        B,T,C,H,W=previous_frames.shape

        previous_frames=previous_frames.reshape(-1,C,H,W) # [B*T, C, H, W]
        # print(f'new shape of previous frames{previous_frames.shape}')

        # apply the base models to extract visual features for each frame
        frame_wise_feature=self.visual_neck2(self.visual_feature2(previous_frames))

        # shape to [B, T, length_of_feature]
        frame_wise_feature=frame_wise_feature.view(B,T,-1)

        # print(f'shape of frame_wise_feature  {frame_wise_feature.shape}')


        visual_feature = self.visual_feature(current_frame)
        visual_feature=self.visual_neck(visual_feature)
        # print(f'shape of visual feature {visual_feature.shape}')

        temporal_context=self.attention(torch.cat((frame_wise_feature,visual_feature.unsqueeze(1)),dim=1))
        fused_feature = self.fuse_block(visual_feature + temporal_context)
        # print(fused_feature.shape)

        # return self.head(fused_feature+visual_feature)

        # return self.head(temporal_context + visual_feature) * torch.tensor([456, 256, 456, 256]).cuda()
        return self.head(torch.cat((fused_feature, visual_feature),dim=-1)) * torch.tensor([456, 256, 456, 256]).cuda()
        # return self.head(temporal_context) * torch.tensor([456, 256, 456, 256])

    # print('shape of visual feature neck',visual_feature.shape)


    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class Attention(nn.Module):
    def __init__(self,dim,num_heads=8,p_drop_att=0.0,p_drop_ffn=0.0,depth=6,time_length=11,hidden_times=3):
        super(Attention,self).__init__()
        self.dim = dim
        self.num_heads=num_heads
        self.pe=PositionalEncoding(d_model=dim)
        self.blocks=nn.ModuleList([
            AttentionBlock(dim=self.dim,num_heads=self.num_heads,p_drop_att=p_drop_att,p_drop_ffn=p_drop_ffn,hidden_times=hidden_times)
            for p in range(depth)
        ])
        self.last_atten=AttentionBlockLast(dim=self.dim,num_heads=self.num_heads,p_drop_att=p_drop_att,p_drop_ffn=p_drop_ffn,hidden_times=hidden_times)
        self.down=nn.Linear(time_length,1)

    def forward(self,x):
        x=x+self.pe(x)
        for block in self.blocks:
            x=block(x)
        x,atten=self.last_atten(x)
        # x.shape= (B, T ,dim)
        # x.tranpose(1,2).shape= (B, dim, T)
        # x=self.down(x.transpose(1,2)).squeeze(2)
        return x,atten

class AttentionBlock(nn.Module):
    def __init__(self,dim,num_heads,p_drop_att=0.0,p_drop_ffn=0.0,hidden_times=2):
        super(AttentionBlock,self).__init__()
        self.dim = dim
        self.num_heads=num_heads
        self.p_drop_att=p_drop_att
        self.p_drop_ffn=p_drop_ffn
        self.hidden_times=hidden_times
        self.multihead_attention=MultiheadAttention(dim=self.dim,num_heads=self.num_heads,p_dropout=self.p_drop_att)
        self.ffn=FFN(dim=self.dim,hidden_times=self.hidden_times,p_dropout=self.p_drop_ffn)

    def forward(self,x):
        return self.ffn(self.multihead_attention(x))


class MultiheadAttention(nn.Module):
    def __init__(self, dim, num_heads, p_dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p_dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, dim = x.shape

        shortcut = x
        x = self.layer_norm(x)
        '''
        x.shape = (B, T, dim)
        self.qkv(x).shape = (B, T, 3*dim)
        self.qkv(x).reshape(B, T, 3, self.num_heads, self.dim // self.num_heads).shape 
            =(B,T,3,num_heads,self.dim// num_heads) 

        self.qkv(x).reshape(B, T, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4).shape
            =(3, B, num_heads, T, self.dim // self.num_heads)
        '''
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        # (B,num_heads,T,head_dim)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        q = q * self.scale

        # k.transpose(-2,-1).shape= (B, num_heads, head_dim, T)
        # atten.shape= (B, num_heads, T, T)
        atten = torch.matmul(q, k.transpose(-2, -1))
        atten = torch.nn.functional.softmax(atten, dim=-1)

        # torch.matmul(atten,v).shape= (B, num_heads, T, head_dim)
        # torch.matmul(atten,v).transpose(1,2).shape= (B, T, num_heads, head_dim)
        # torch.matmul(atten,v).transpose(1,2).reshape(B, T, -1).shape= (B, T, dim)
        z = torch.matmul(atten, v).transpose(1, 2).reshape(B, T, -1)
        z = shortcut + z
        return z


class AttentionBlockLast(nn.Module):
    def __init__(self,dim,num_heads,p_drop_att=0.0,p_drop_ffn=0.0,hidden_times=2):
        super(AttentionBlockLast,self).__init__()
        self.dim = dim
        self.num_heads=num_heads
        self.p_drop_att=p_drop_att
        self.p_drop_ffn=p_drop_ffn
        self.hidden_times=hidden_times
        self.multihead_attention=MultiheadAttentionLast(dim=self.dim,num_heads=self.num_heads,p_dropout=self.p_drop_att)

    def forward(self,x):
        z,atten=self.multihead_attention(x)
        return z,atten



class MultiheadAttentionLast(nn.Module):
    def __init__(self, dim, num_heads, p_dropout=0.0):
        super(MultiheadAttentionLast, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.softmax = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(p_dropout)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, dim = x.shape

        shortcut = x
        x = self.layer_norm(x)
        '''
        x.shape = (B, T, dim)
        self.qkv(x).shape = (B, T, 3*dim)
        self.qkv(x).reshape(B, T, 3, self.num_heads, self.dim // self.num_heads).shape 
            =(B,T,3,num_heads,self.dim// num_heads) 

        self.qkv(x).reshape(B, T, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4).shape
            =(3, B, num_heads, T, self.dim // self.num_heads)
        '''
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        # (B,num_heads,T,head_dim)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        q = q * self.scale

        # k.transpose(-2,-1).shape= (B, num_heads, head_dim, T)
        # atten.shape= (B, num_heads, T, T)
        atten = torch.matmul(q, k.transpose(-2, -1))
        atten = torch.nn.functional.softmax(atten, dim=-1)

        # torch.matmul(atten,v).shape= (B, num_heads, T, head_dim)
        # torch.matmul(atten,v).transpose(1,2).shape= (B, T, num_heads, head_dim)
        # torch.matmul(atten,v).transpose(1,2).reshape(B, T, -1).shape= (B, T, dim)
        z = torch.matmul(atten, v).transpose(1, 2).reshape(B, T, -1)
        z = shortcut + z
        # print(f'shape of z {z.shape}')
        return z[:,-1,:], atten[:,:,-1,:]


class FFN(nn.Module):
    def __init__(self,dim,hidden_times=2,p_dropout=0.1):
        super(FFN,self).__init__()
        self.dim=dim
        self.hidden_times=hidden_times
        self.p_dropout=p_dropout
        self.layer_norm=nn.LayerNorm(self.dim)
        self.ffn=nn.Sequential(
            nn.Linear(self.dim,self.hidden_times*self.dim),
            nn.Dropout(self.p_dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_times*self.dim,self.dim)
        )
    def forward(self,x):
        return x+self.ffn(self.layer_norm(x))


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


if __name__=='__main__':


    # x=torch.rand(2,11,512)
    # attention=Attention(dim=512,num_heads=8,depth=6)
    # print(attention(x).shape)

    model=IntentNetDataAttention()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'model size: {total_params}')
    frames = torch.rand(2, 11, 3, 224, 224)
    # output=model(frames)
    output,atten = model(frames)
    # outputs_history=models(previous_frames)
    # print(outputs_history.shape)
    print(output.shape)
    print(atten.shape)

