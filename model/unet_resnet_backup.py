
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



resnet18 = models.resnet18(pretrained=True)


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3,
                 stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding,
                              kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpsampleBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of
    Upsample -> ConvBlock -> ConvBlock
    """
    def __init__(self, in_channels, out_channels,
                 up_conv_in_channels=None,
                 up_conv_out_channels=None,
                 ):
        super(UpsampleBlock, self).__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels
        self.upsample = nn.ConvTranspose2d(up_conv_in_channels,
                                               up_conv_out_channels,
                                               kernel_size=2,
                                               stride=2)

        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        # print(f'shape of x before upsample: {up_x.shape}')
        x = self.upsample(up_x)

        # print(f'shape of x after upsample: {x.shape}')
        x = torch.cat([x, down_x], 1)
        # print(x.shape)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x





class UNetResNet18(nn.Module):
    def __init__(self, n_classes=2):
        super(UNetResNet18, self).__init__()
        self.DEPTH = 6
        down_blocks = []
        up_blocks = []

        self.input_block = nn.Sequential(*list(resnet18.children()))[:3]  #
        self.input_pool = list(resnet18.children())[3]  # MaxPool2d

        for bottleneck in list(resnet18.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(512, 512)

        # up_blocks
        up_blocks.append(UpsampleBlock(512, 256))
        up_blocks.append(UpsampleBlock(256, 128))
        up_blocks.append(UpsampleBlock(128, 64))
        up_blocks.append(UpsampleBlock(in_channels=64 + 64,
                                       out_channels=128,
                                       up_conv_in_channels=64,
                                       up_conv_out_channels=64))
        up_blocks.append(UpsampleBlock(in_channels=64 + 3,
                                       out_channels=64,
                                       up_conv_in_channels=128,
                                       up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        # print(f'shape of riginal x: {x.shape}')
        pre_pools = dict()
        pre_pools[f'layer_0'] = x
        x = self.input_block(x)
        # print(f'shape of x after input_block: {x.shape}')
        pre_pools[f'layer_1'] = x
        x = self.input_pool(x)

        # print(f'shape of x after input_pool: {x.shape}')

        for i, block in enumerate(self.down_blocks, start=2):
            x = block(x)
            if i == (self.DEPTH-1):
                continue
            pre_pools[f'layer_{i}'] = x

        # print(f'shape of x after down block: {x.shape}')
        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, start=1):
            # print(f'shape of x before upblock: {x.shape}')
            key = f'layer_{self.DEPTH - 1 - i}'
            # print(f'shape of key {pre_pools[key].shape}')
            x = block(x, pre_pools[key])

        output_feature_map = x

        x = self.out(x)
        del pre_pools

        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x








if __name__ == '__main__':
    from torchsummary import summary
    # model = resnet34()
    # summary(your_model, input_size=(channels, H, W))
    model = UNetResNet18()
    # model = UNetVGG16()
    # model = ResNetEncoderDecoder()
    # model = UNetResNet18AdlDrop()
    # summary(model.cuda(), input_size=(3, 512, 512))
    img=torch.rand((2,3, 224, 224))
    output=model(img)
    print(f'shape of output: {output.shape}')
    # summary(model, input_size=(3, 224, 224))  # 被32整除

