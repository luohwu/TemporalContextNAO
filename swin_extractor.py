import argparse

import torch


from config import get_config
from models.build import build_model

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg',default='/cluster/home/luohwu/workspace/TemporalContextNAO/configs/swin_tiny_patch4_window7_224.yaml'
                        , type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume',default='checkpoints/swin_tiny_patch4_window7_224.pth', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int,default=1, required=False, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

class SwinTransformer(torch.nn.Module):
    def __init__(self):
        super(SwinTransformer,self).__init__()
        args,config=parse_option()
        model = build_model(config)
        modules = list(model.children())
        # for i in range(len(modules)):
        #     print('='*50)
        #     print(modules[i])
        self.down=torch.nn.Sequential(*modules[:2])
        self.extrator=torch.nn.Sequential(*modules[2])
        self.down2=torch.nn.Sequential(*modules[3:5])
        # self.MLP=torch.nn.Sequential(
        #     torch.nn.Linear(49,512),
        #     torch.nn.ReLU(512)
        # )


    def forward(self,x):
        x=self.down(x)
        x=self.extrator(x)
        x=self.down2(x).squeeze(2)
        # return self.MLP(x)
        return x

if __name__=='__main__':


    x=torch.rand(2,3,224,224)
    model=SwinTransformer()
    y=model(x)
    print(y.shape)
