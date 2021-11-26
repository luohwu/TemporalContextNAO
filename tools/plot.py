import torch
import matplotlib.pyplot as plt
exp_name='ADL_4gpus'
loss=torch.load(f'/mnt/euler/experiments/{exp_name}/ckpts/loss.pth')
plt.plot(loss['train_loss'])
plt.plot(loss['val_loss'])
plt.legend(['train loss', 'val loss'])
title='Baseline bbox'
plt.title(title)
plt.savefig(f'/home/luohwu/Thesis_workspace/Figures/{title}.jpg')