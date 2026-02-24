import torch
import torch.nn as nn
import torchvision

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.residual_connection = nn.Sequential()
        if in_channels != out_channels:
            self.residual_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # ❌ 不加载预训练参数
        self.img_encoder = torchvision.models.resnet18(pretrained=True)

        self.img_encoder = nn.Sequential(*list(self.img_encoder.children())[:-2])  # 去掉最后两层（全局池化 + 分类层）

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.residual_block = ResidualBlock(512, 512)

    def forward(self, x):
        x = self.img_encoder(x)
        x = self.global_avg_pool(x)
        x = self.residual_block(x)
        x = x.view(x.size(0), -1)  # 输出 [batch_size, 512]
        return x
