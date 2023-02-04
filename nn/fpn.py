'''
    Includes Feature Pyramid Network & Class/Box Subnets.
'''
from torch import nn
from utils import padding


class FPN(nn.Module):
    def __init__(self, input_sizes, input_channels=(512, 1024, 2048)):
        super().__init__()
        c3_input_size, c4_input_size, c5_input_size = input_sizes
        c3_input_channels, c4_input_channels, c5_input_channels = input_channels
        self.c5_conv1 = nn.Conv2d(c5_input_channels, 256, kernel_size=1, padding="same")
        self.c5_upsample = nn.Upsample(size=c4_input_size, mode="bilinear")
        self.c5_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.c5_conv3 = nn.Conv2d(
            c5_input_channels,
            256,
            kernel_size=1,
            stride=2,
            padding=padding(c4_input_channels, 1, 2),
        )
        self.c5_conv4 = nn.Conv2d(
            256,
            256,
            kernel_size=1,
            stride=2,
            padding=padding(c4_input_channels, 1, 2),
        )
        self.c4_conv1 = nn.Conv2d(c4_input_channels, 256, kernel_size=1, padding="same")
        self.c4_upsample = nn.Upsample(size=c3_input_size, mode="bilinear")
        self.c4_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding="same")
        self.c3_conv1 = nn.Conv2d(c3_input_channels, 256, kernel_size=1, padding="same")
        self.c3_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding="same")

    def forward(self, inputs):
        c3, c4, c5 = inputs
        p6 = self.c5_conv3(c5)
        p7 = self.c5_conv4(p6)
        c5_conv = self.c5_conv1(c5)
        p5 = self.c5_conv2(c5_conv)
        c5_c4 = self.c5_upsample(c5_conv) + self.c4_conv1(c4)
        p4 = self.c4_conv2(c5_c4)
        c4_c3 = self.c4_upsample(c5_c4) + self.c3_conv1(c3)
        p3 = self.c3_conv2(c4_c3)
        return p3, p4, p5, p6, p7
    

if __name__ == '__main__':
    from resnet import ResNet50
    import torch
    x = torch.rand((16, 3, 400, 400)).to('cuda')
    resnet = ResNet50().to('cuda')
    resnet_output = resnet(x)
    fpn = FPN([c.shape[-1] for c in resnet_output]).to('cuda')
    print([p.shape for p in fpn(resnet_output)])