from comet_ml import Experiment
from datetime import datetime
import matplotlib.pyplot as plt
import torch.cuda
from torch import optim
from torch.utils.data import DataLoader
from data.dataset import *
from opt import *
import tarfile
from tools.CIOU import CIOU_LOSS,CIOU_LOSS2,cal_acc_f1
from model.temporal_context_net import IntentNet,IntentNetSwin,IntentNetFuse,IntentNetIC,IntentNetFuseAttention
from torch import  nn
import pandas as pd
import cv2
from tools.comparison import generate_comparison
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
experiment = Experiment(
    api_key="wU5pp8GwSDAcedNSr68JtvCpk",
    project_name="intent-net-with-temporal-context",
    workspace="thesisproject",
    auto_metric_logging=False
)
experiment.log_code(file_name="model/temporal_context_net.py")
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


def main():
    # model=IntentNet()

    # model=IntentNetSwin(time_length=10)
    # for p in model.temporal_context_extractor.parameters():
    #     p.requires_grad=False

    # model=IntentNetFuse()
    model=IntentNetFuseAttention()
    # model = IntentNetIC()
    # for p in model.temporal_context.parameters():
    #     p.requires_grad=False
    # cnt=0
    # for child in model.temporal_context.children():
    #     cnt+=1
    #     if cnt<=4:
    #         for p in child.parameters():
    #             p.requires_grad=False

    # for p in model.temporal_context_extractor.parameters():
    #     p.requires_grad = False

    if multi_gpu == True:
        model = nn.DataParallel(model)
    model = model.to(device)

    # if args.original_split:
    #     train_dataset = NAODataset(mode='train', dataset_name=args.dataset)
    #     val_dataset = NAODataset(mode='val', dataset_name=args.dataset)
    # else:
    #     all_data = NAODataset(mode='all', dataset_name=args.dataset)
    #     if args.dataset == 'ADL':
    #         train_dataset, val_dataset = torch.utils.data.random_split(all_data, [1767, 450])
    #     else:
    #         train_dataset, val_dataset = torch.utils.data.random_split(all_data, [8589, 3000])
    #
    train_dataset, val_dataset = ini_datasets(dataset_name=args.dataset, original_split=args.original_split)



    print(f'train dataset size: {len(train_dataset)}, val dataset size: {len(val_dataset)}')
    train_dataloader = DataLoader(train_dataset, batch_size=args.bs,
                                  shuffle=True, num_workers=4,
                                  pin_memory=True,
                                  drop_last=True if torch.cuda.device_count() >=4 else False)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.bs,
                                shuffle=True, num_workers=4,
                                pin_memory=True,
                                drop_last=True if torch.cuda.device_count() >= 4 else False)

    if args.SGD:
        print('using SGD')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:

        print('using AdamW')
        optimizer = optim.AdamW(filter(lambda  p: p.requires_grad,model.parameters()),
                                lr=args.lr,
                                betas=(0.9, 0.99)
                                # ,weight_decay=args.weight_decay
                                )





    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.8,
                                                     patience=3,
                                                     verbose=True,
                                                     min_lr=0.00001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=4e-5,verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=25,eta_min=1e-5,verbose=True)
    # scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.98,verbose=True)

    """"
    Heatmap version
    """

    criterion = CIOU_LOSS()
    # criterion=nn.MSELoss()

    train_args['ckpt_path'] = os.path.join(train_args['exp_path'],args.dataset,
                                           exp_name, 'ckpts/')
    if not os.path.exists(train_args['ckpt_path']):
        os.mkdir(train_args['ckpt_path'])


    train_loss_list = []
    val_loss_list = []
    current_epoch = 0
    epoch_save = 50 if args.dataset == 'EPIC' else 50
    for epoch in range(current_epoch + 1, train_args['epochs'] + 1):
        print(f"==================epoch :{epoch}/{train_args['epochs']}===============================================")

        train_loss = train(train_dataloader, model, criterion, optimizer, epoch=epoch)
        val_loss = val(val_dataloader, model, criterion, epoch, illustration=False)
        # scheduler.step(val_loss)
        scheduler.step()
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if epoch % epoch_save == 0:
            checkpoint_path = os.path.join(train_args['ckpt_path'], f'model_epoch_{epoch}.pth')

            # torch.save({'epoch': epoch,
            #             'model_state_dict': model.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict()
            #             },
            #            checkpoint_path)
            val(val_dataloader, model, criterion, epoch, illustration=True)

        experiment.log_metrics({"val_loss": val_loss, "train_loss": train_loss}, step=epoch)
        print(f'train loss: {train_loss:.8f} | val loss:{val_loss:.8f}')


def train(train_dataloader, model, criterion, optimizer,epoch):
    train_losses = 0.
    total_acc=0
    total_f1=0

    len_dataset = len(train_dataloader.dataset)
    for i, data in enumerate(train_dataloader, start=1):
        previous_frames,current_frame, nao_bbox, img_path = data
        previous_frames=previous_frames.to(device)
        current_frame=current_frame.to(device)
        nao_bbox=nao_bbox.to(device)

        #forward
        outputs = model(previous_frames,current_frame)
        del previous_frames,current_frame

        # loss and acc
        loss, acc,f1,_ = criterion(outputs, nao_bbox)
        # loss = criterion(outputs, nao_bbox)
        # acc, f1, conf_matrix = cal_acc_f1(outputs, nao_bbox)


        del outputs, nao_bbox

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += loss.item()
        total_f1 += f1.sum().item()
        total_acc += acc.sum().item()
    acc_avg = total_acc / len_dataset
    f1_avg=total_f1/len_dataset
    experiment.log_metric("train_acc_avg", acc_avg, step=epoch)
    experiment.log_metric("train_f1_avg", f1_avg, step=epoch)

    return train_losses / len_dataset
    # return train_losses

def val(val_dataloader, model, criterion, epoch, illustration):
    model.eval()
    total_val_loss = 0
    total_acc=0
    total_f1=0
    total_conf_matrix=np.zeros([2,2])
    global_min_acc=999
    global_max_acc=-999
    len_dataset = len(val_dataloader.dataset)
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            previous_frames,current_frame,nao_bbox, img_path = data
            previous_frames=previous_frames.to(device)
            current_frame=current_frame.to(device)
            nao_bbox=nao_bbox.to(device)

            outputs = model(previous_frames,current_frame)
            del previous_frames,current_frame


            loss, acc,f1,conf_matrix = criterion(outputs, nao_bbox)
            # loss = criterion(outputs, nao_bbox)
            # acc, f1, conf_matrix=cal_acc_f1(outputs, nao_bbox)


            total_val_loss += loss.item()
            total_f1+=f1.sum().item()
            total_acc += acc.sum().item()
            total_conf_matrix+=conf_matrix


            if illustration:
                min_acc,min_index=f1.min(0) ###########################################
                max_acc,max_index=f1.max(0) ##############################################
                if global_min_acc > min_acc:
                    global_min_acc = min_acc
                    global_min_image = img_path[min_index]
                    global_min_nao_bbox_gt = nao_bbox[min_index]
                    global_min_nao_bbox = outputs[min_index]
                if global_max_acc < max_acc:
                    global_max_acc = max_acc
                    global_max_image = img_path[max_index]
                    global_max_nao_bbox_gt = nao_bbox[max_index]
                    global_max_nao_bbox = outputs[max_index]
            del outputs, nao_bbox

        if illustration:
            if global_min_acc<999:
                illustration_worst = generate_comparison(global_min_image, global_min_nao_bbox, global_min_nao_bbox_gt)
                experiment.log_image(illustration_worst, name=f'worst_{epoch}',step=epoch)
                experiment.log_text(f'worst_{epoch}: {global_min_image}',step=epoch)
            if global_max_acc>-999:
                illustration_best=generate_comparison(global_max_image, global_max_nao_bbox, global_max_nao_bbox_gt)
                experiment.log_image(illustration_best, name=f'best_{epoch}',step=epoch)
                experiment.log_text(f'best_{epoch}: {global_max_image}', step=epoch)

        val_loss_avg = total_val_loss / len_dataset
        acc_avg = total_acc / len_dataset
        f1_avg=total_f1/len_dataset
        conf_matrix_avg=(total_conf_matrix/len_dataset).astype(np.int32)
    print(f'[epoch {epoch}], [val loss {val_loss_avg:5f}], [acc avg {acc_avg:5f}],[f1 avg {f1_avg:5f}] ')
    experiment.log_metric("val_acc_avg", acc_avg, step=epoch)
    experiment.log_metric("val_f1_avg", f1_avg, step=epoch)
    experiment.log_confusion_matrix(matrix=conf_matrix_avg, title=f"confusion matrix epoch {epoch}",
                                    file_name=f"confusion_matrix_epoc_{epoch}.json",row_label="Actual Category",
                                    column_label="Predicted Category",labels=["1","0"],step=epoch)

    model.train()

    return val_loss_avg



if __name__ == '__main__':

    if args.euler:
        scratch_path = os.environ['TMPDIR']
        tar_path = '/cluster/home/luohwu/dataset.tar.gz'
        assert os.path.exists(tar_path), f'file not exist: {tar_path}'
        print('extracting dataset from tar file')
        tar = tarfile.open(tar_path)
        tar.extractall(os.environ['TMPDIR'])
        tar.close()
        print('finished')
    # train_data = EpicDatasetV2('train')

    main()

