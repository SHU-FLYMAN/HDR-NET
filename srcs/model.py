import torch.nn as nn
import torch.nn.functional as F
import torch


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        y = x
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.bn2(self.conv2(y))

        # down-sample of X
        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNetFPN(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Config
        block = BasicBlock
        initial_dim = 128
        block_dims = [128, 196, 256]

        self.conv_128_1 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_128_2 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1, bias=False)

        #  --------- 01 CNN1 --------- #
        # Class Variable
        self.in_planes = initial_dim

        # Networks
        self.conv1_1 = nn.Conv2d(in_channels, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1_1 = nn.BatchNorm2d(initial_dim)
        self.relu_1 = nn.ReLU(inplace=True)

        self.layer1_1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2_1 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3_1 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv_1 = conv1x1(block_dims[2], block_dims[2])

        self.layer2_outconv_1 = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2_1 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),

            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv_1 = conv1x1(block_dims[0], block_dims[1])

        self.layer1_outconv2_1 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        #  --------- 02 CNN2 --------- #
        # Class Variable
        # Networks
        self.in_planes = initial_dim

        self.conv1_2 = nn.Conv2d(in_channels + 1, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1_2 = nn.BatchNorm2d(initial_dim)
        self.relu_2 = nn.ReLU(inplace=True)

        self.layer1_2 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2_2 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3_2 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        # 3. FPN upsample
        self.layer3_outconv_2 = conv1x1(block_dims[2], block_dims[2])
        self.layer2_outconv_2 = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2_2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv_2 = conv1x1(block_dims[0], block_dims[1])
        self.layer1_outconv2_2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )
        # --------- 03 init model --------- #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, input_img):
        # CNN1
        # ResNet Backbone
        # 3x1248x1632
        x0 = self.relu_1(self.bn1_1(self.conv1_1(input_img)))  # 128 x 624 x 816

        x1 = self.layer1_1(x0)  # 1/2  # 128 x 624 x 816
        x2 = self.layer2_1(x1)  # 1/4  # 196 x 312 x 408
        x3 = self.layer3_1(x2)  # 1/8  # 256 x 156 x 204

        # FPN
        x3_out = self.layer3_outconv_1(x3)  # 256 -> 256 x 156 x 204

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)   # 256 x 312 x 408 上采样

        x2_out = self.layer2_outconv_1(x2)                   # 196 -> 256 x 312 x 408
        x2_out = self.layer2_outconv2_1(x2_out + x3_out_2x)  # 256 -> 196 x 312 x 408 直接加法
        x2_out_1x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)  # 196 x 624 x 816

        x1_out = self.layer1_outconv_1(x1)                   # 128 -> 196 x 624 x 816
        x1_out = self.layer1_outconv2_1(x1_out + x2_out_1x)  # 196 -> 128 x 624 x 816

        # output of CNN1: 1 x 1248x1632
        cnn1_out = self.conv_128_1(F.interpolate(x1_out, scale_factor=2., mode='bilinear', align_corners=True))

        # CNN2
        input2 = torch.concat((input_img, cnn1_out), 1)  # 4 x 1248 x 1632

        x0_2 = self.relu_2(self.bn1_2(self.conv1_2(input2)))  # 128 x 624 x 816
        x1_2 = self.layer1_2(x0_2)  # 1/2   # 128 x 624 x 816
        x2_2 = self.layer2_2(x1_2)  # 1/4   # 128 -> 196 x 312 x 408
        x3_2 = self.layer3_2(x2)    # 1/8   # 196 -> 256 x 156 x 204

        # # FPN
        x3_out_2 = self.layer3_outconv_2(x3_2)  # 256 x 156 x 204
        x3_out_2x_2 = F.interpolate(x3_out_2, scale_factor=2., mode='bilinear', align_corners=True)  # 256 x 312 x 408

        x2_out_2 = self.layer2_outconv_2(x2_2)  # 196 -> 256 x 312 x 408
        x2_out_2 = self.layer2_outconv2_2(x2_out_2 + x3_out_2x_2)  # 256 -> 196 x 312 x 408

        x2_out_1x_2 = F.interpolate(x2_out_2, scale_factor=2., mode='bilinear', align_corners=True)  # 196 x 624 x 816

        x1_out_2 = self.layer1_outconv_2(x1_2)  # 128 -> 196 x 624 x 816
        x1_out_2 = self.layer1_outconv2_2(x1_out_2 + x2_out_1x_2)   # 196 -> 128 x 624 x 816

        # 2 x 1248 x 1632
        cnn2_out_2 = self.conv_128_2(F.interpolate(x1_out_2, scale_factor=2., mode='bilinear', align_corners=True))

        return cnn1_out, cnn2_out_2


# 普通网络
class UNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        block = BasicBlock
        initial_dim = 128
        block_dims = [128, 196, 256]

        self.conv_128_1 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.in_planes = initial_dim

        # Networks
        self.conv1_1 = nn.Conv2d(in_channels, initial_dim, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1_1 = nn.BatchNorm2d(initial_dim)
        self.relu_1 = nn.ReLU(inplace=True)

        self.layer1_1 = self._make_layer(block, block_dims[0], stride=1)  # 1/2
        self.layer2_1 = self._make_layer(block, block_dims[1], stride=2)  # 1/4
        self.layer3_1 = self._make_layer(block, block_dims[2], stride=2)  # 1/8

        self.layer3_outconv_1 = conv1x1(block_dims[2], block_dims[2])

        self.layer2_outconv_1 = conv1x1(block_dims[1], block_dims[2])
        self.layer2_outconv2_1 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),

            conv3x3(block_dims[2], block_dims[1]),
        )
        self.layer1_outconv_1 = conv1x1(block_dims[0], block_dims[1])

        self.layer1_outconv2_1 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0]),
        )

        # init model
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, dim, stride=1):
        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, input_img):
        x0 = self.relu_1(self.bn1_1(self.conv1_1(input_img)))  # 128 x 624 x 816
        x1 = self.layer1_1(x0)  # 1/2  # 128 x 624 x 816
        x2 = self.layer2_1(x1)  # 1/4  # 196 x 312 x 408
        x3 = self.layer3_1(x2)  # 1/8  # 256 x 156 x 204

        x3_out = self.layer3_outconv_1(x3)  # 256 -> 256 x 156 x 204

        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)  # 256 x 312 x 408 上采样

        x2_out = self.layer2_outconv_1(x2)  # 196 -> 256 x 312 x 408
        x2_out = self.layer2_outconv2_1(x2_out + x3_out_2x)  # 256 -> 196 x 312 x 408 直接加法
        x2_out_1x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)  # 196 x 624 x 816

        x1_out = self.layer1_outconv_1(x1)  # 128 -> 196 x 624 x 816
        x1_out = self.layer1_outconv2_1(x1_out + x2_out_1x)  # 196 -> 128 x 624 x 816

        # 网络1的输出 2 x 1248x1632
        cnn1_out = self.conv_128_1(F.interpolate(x1_out, scale_factor=2., mode='bilinear', align_corners=True))
        return cnn1_out

def cnn1_loss(Y, ground_img):
    loss = nn.L1Loss(reduction="mean")
    return loss(Y, ground_img)


def cnn2_loss(Y, numerator, denominator):
    loss_numerator = nn.L1Loss(reduction="mean")
    loss_denominator = nn.L1Loss(reduction="mean")

    # NCHW
    Y_n, Y_d = torch.split(Y, 1, dim=1)

    l_num = loss_numerator(Y_n, numerator)
    l_den = loss_denominator(Y_d, denominator)
    return l_num + l_den


if __name__ == '__main__':
    import onnx     # test model
    save_file = "../logs/model.onnx"
    h = 1248
    w = 1632
    data = torch.randn(2, 3, h, w)
    net = ResNetFPN()
    torch.onnx.export(
        net, data, save_file,
        export_params=False
    )
    print(net)
    onex_model = onnx.load(save_file)
    onnx.save(onnx.shape_inference.infer_shapes(onex_model), save_file)