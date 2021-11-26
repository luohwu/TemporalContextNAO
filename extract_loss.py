import torch
import matplotlib.pyplot as plt
exp_name='ADL_4gpus'
epoch=800
data_path=f'/cluster/home/luohwu/experiments/{exp_name}/ckpts'
dict=torch.load(f'{data_path}/model_epoch_{epoch}.pth',map_location='cpu')
train_loss=dict['train_loss_list']
val_loss=dict['val_loss_list']
torch.save({'train_loss':train_loss,
            'val_loss': val_loss},
           f'{data_path}/loss.pth')
