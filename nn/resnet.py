'''
    Includes a ResNet 50.
'''
from torch import nn
from utils import padding

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=1, bottleneck=True):
        super().__init__()
        if not bottleneck and in_channels != out_channels:
            raise ValueError(
                "When Resblock is not a bottleneck, output channels should match input channels"
            )
        if bottleneck:
            conv_channels = out_channels // 4
        else:
            conv_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=1, stride=stride, padding=padding(in_channels, 1, stride)),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(conv_channels),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                conv_channels,
                out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_res = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding(in_channels, 1, stride),
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.LeakyReLU(0.2)
        self.bottleneck = bottleneck

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.bottleneck:
            conv_res = self.conv_res(input)
            return self.relu(conv_res + x)
        else:
            return self.relu(input + x)


class Conv1(nn.Module):
    """
    ResNet input block, transforms input shape(B, C, W, H) -> (B, 64, W/4, H/4)
    """

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=padding(in_channels, 7, 2),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2(nn.Module):
    def __init__(self, in_channels=64, out_channels=256, layer_nums=3):
        super().__init__()
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=padding(in_channels, 3, 2)
        )
        self.bottleneck = ResBlock(in_channels, out_channels)
        self.layers = nn.Sequential(
            *[
                ResBlock(out_channels, out_channels, bottleneck=False)
                for _ in range(layer_nums - 1)
            ]
        )

    def forward(self, input):
        x = self.maxpool(input)
        x = self.bottleneck(x)
        return self.layers(x)


class Conv3(nn.Module):
    def __init__(self, in_channels=256, out_channels=512, layer_nums=4):
        super().__init__()
        self.bottleneck = ResBlock(in_channels, out_channels, stride=2)
        self.layers = nn.Sequential(
            *[
                ResBlock(out_channels, out_channels, bottleneck=False)
                for _ in range(layer_nums - 1)
            ]
        )

    def forward(self, input):
        x = self.bottleneck(input)
        return self.layers(x)


class Conv4(nn.Module):
    def __init__(self, in_channels=512, out_channels=1024, layer_nums=6):
        super().__init__()
        self.bottleneck = ResBlock(in_channels, out_channels, stride=2)
        self.layers = nn.Sequential(
            *[
                ResBlock(out_channels, out_channels, bottleneck=False)
                for _ in range(layer_nums - 1)
            ]
        )

    def forward(self, input):
        x = self.bottleneck(input)
        return self.layers(x)


class Conv5(nn.Module):
    def __init__(self, in_channels=1024, out_channels=2048, layer_nums=3):
        super().__init__()
        self.bottleneck = ResBlock(in_channels, out_channels, stride=2)
        self.layers = nn.Sequential(
            *[
                ResBlock(out_channels, out_channels, bottleneck=False)
                for _ in range(layer_nums - 1)
            ]
        )

    def forward(self, input):
        x = self.bottleneck(input)
        return self.layers(x)


class ResNet50(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = Conv1(in_channels)
        self.conv2 = Conv2()
        self.conv3 = Conv3()
        self.conv4 = Conv4()
        self.conv5 = Conv5()

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        return c3, c4, c5


if __name__ == "__main__":
    import torch

    input = torch.rand((16, 3, 300, 300)).to("cuda")
    resnet = ResNet50().to("cuda")
    c3, c4, c5 = resnet(input)
    print(c3.shape, c4.shape, c5.shape)
