import torch
import matplotlib.pyplot as plt
import pandas as pd
exp_name='temporal_ADL_bbox_3e-4'
# exp_name='ADL_4gpus'

# loss=torch.load(f'/mnt/euler/experiments/{exp_name}/ckpts/loss.pth')
loss=pd.read_csv((f'/mnt/euler/experiments/{exp_name}/ckpts/loss.csv'))
plt.plot(loss['train_loss'])
plt.plot(loss['val_loss'])
plt.legend(['train loss', 'val loss'])
title='IntentNet bbox'
plt.title(title)
plt.savefig(f'/home/luohwu/Thesis_workspace/Figures/{title}.png')
plt.show()
