
from datetime import datetime

import torch.cuda
from torch import optim
from torch.utils.data import DataLoader
from data.dataset import NAODataset
from opt import *
import tarfile
from tools.CIOU import CIOU_LOSS
from model.temporal_context_net import TemporalNaoNet
from torch import  nn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

exp_name = args.exp_name

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


multi_gpu = True if torch.cuda.device_count() > 1 else False
print(f'using {torch.cuda.device_count()} GPUs')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    model=TemporalNaoNet(time_length=10)
    for p in model.temporal_context_extractor.parameters():
        p.requires_grad = False

    if multi_gpu == True:
        model = nn.DataParallel(model)
    model = model.to(device)
    train_data=NAODataset(mode='train',dataset_name='ADL')
    train_dataloader = DataLoader(train_data, batch_size=args.bs,
                                  shuffle=True, num_workers=2,
                                  pin_memory=True)
    val_data = NAODataset(mode='val',dataset_name='ADL')
    val_dataloader = DataLoader(val_data,
                                batch_size=args.bs,
                                shuffle=True, num_workers=2,
                                pin_memory=True)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.99),
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.8,
                                                     patience=5,
                                                     verbose=True,
                                                     min_lr=0.0000001)

    """"
    Heatmap version
    """

    # if args.dataset == 'EPIC':
    #     # class_weights = torch.FloatTensor([1, 11.2]).cuda(args.device_ids[0])
    #     class_weights = torch.FloatTensor([1, 11.2]).to(device)
    # else:
    #     # class_weights = torch.FloatTensor([1, 9.35]).cuda(args.device_ids[0])
    #     class_weights = torch.FloatTensor([1, 9.35]).to(device)
    # criterion = nn.CrossEntropyLoss(class_weights)
    criterion = CIOU_LOSS()
    # criterion = FocalLoss()

    train_args['ckpt_path'] = os.path.join(train_args['exp_path'],
                                           exp_name, 'ckpts/')
    if not os.path.exists(train_args['ckpt_path']):
        os.mkdir(train_args['ckpt_path'])

    write_val = open(os.path.join(train_args['ckpt_path'], 'val.txt'), 'w')

    train_loss_list = []
    val_loss_list = []
    current_epoch = 0
    epoch_save = 50 if args.dataset == 'EPIC' else 200
    for epoch in range(current_epoch + 1, train_args['epochs'] + 1):
        print(f"==================epoch :{epoch}/{train_args['epochs']}===============================================")
        train_loss = train(train_dataloader, model, criterion, optimizer, epoch, train_args)
        val_loss = val(val_dataloader, model, criterion, epoch, write_val)
        scheduler.step(val_loss)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if epoch % epoch_save == 0:
            checkpoint_path = os.path.join(train_args['ckpt_path'], f'model_epoch_{epoch}.pth')
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss_list': val_loss_list,
                        'train_loss_list': train_loss_list},
                       checkpoint_path)
        print(f'train loss: {train_loss:.8f} | val loss:{val_loss:.8f}')


def train(train_dataloader, model, criterion, optimizer, epoch, train_args):
    train_losses = 0.

    for i, data in enumerate(train_dataloader, start=1):
        previous_frames,current_frame, nao_bbox = data
        previous_frames=previous_frames.to(device)
        current_frame=current_frame.to(device)
        nao_bbox=nao_bbox.to(device)

        #forward
        outputs = model(previous_frames,current_frame)
        del previous_frames,current_frame

        loss, _ = criterion(outputs, nao_bbox.to(device))

        del outputs, nao_bbox

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses += loss.item()

    return train_losses / len(train_dataloader.dataset)


def val(val_dataloader, model, criterion, epoch, write_val):
    model.eval()
    total_val_loss = 0
    total_iou = 0
    num_correct = 0
    iou_threshold = 0.5
    len_dataset = len(val_dataloader.dataset)
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            # print(f'{i}/{loader_size}')
            previous_frames,current_frame,nao_bbox = data
            previous_frames=previous_frames.to(device)
            current_frame=current_frame.to(device)
            nao_bbox=nao_bbox.to(device)

            outputs = model(previous_frames,current_frame)
            del previous_frames,current_frame
            loss, iou = criterion(outputs, nao_bbox)
            num_correct += torch.sum(iou > iou_threshold).item()
            total_val_loss += loss.item()
            total_iou += torch.sum(iou).item()
            del outputs, nao_bbox

    val_loss_avg = total_val_loss / len_dataset
    iou_avg = total_iou / len_dataset
    print(
        f'[epoch {epoch}], [val loss {val_loss_avg:5f}], [IOU avg {iou_avg:5f}], [acc avg {num_correct / len_dataset}]')

    write_val.writelines(f"[epoch {epoch}], [IOU avg {iou_avg:5f}],[acc avg {num_correct / len_dataset}] \n")


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