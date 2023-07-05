import torch
import torch.nn as nn
import torch.nn.functional as F


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(198, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = x.reshape(-1, x.shape[-1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ConvStack(nn.Module):
    def __init__(self, n_mels, cnn_unit, fc_unit):
        super().__init__()

        # shape of input: (batch_size, 1 channel, frames, input_features)
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(cnn_unit, cnn_unit, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(cnn_unit, cnn_unit * 2, (3, 3), padding=1),
            nn.BatchNorm2d(cnn_unit * 2),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(nn.Linear((cnn_unit * 2) * (n_mels // 4), fc_unit), nn.Dropout(0.5))

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, base_width=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    r"""
    3x3 convolution with padding
    - in_planes: in_channels
    - out_channels: out_channels
    - bias=False: BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4  # 블록 내에서 차원을 증가시키는 3번째 conv layer에서의 확장계수

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # ResNext나 WideResNet의 경우 사용
        width = int(planes * (base_width / 64.0)) * groups

        # Bottleneck Block의 구조
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # conv2에서 downsample
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # 1x1 convolution layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3x3 convolution layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # 1x1 convolution layer
        out = self.conv3(out)
        out = self.bn3(out)
        # skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, block_channel_starts, n_cls=5, input_kernel_size=3):
        super(ResNet, self).__init__()

        self.in_planes = block_channel_starts

        self.conv1 = nn.Conv2d(input_kernel_size, block_channel_starts, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(block_channel_starts)
        self.layer1 = self._make_layer(block, block_channel_starts, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, block_channel_starts * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, block_channel_starts * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, block_channel_starts * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(block_channel_starts * 8 * block.expansion, n_cls)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def net_wrapper(net, **kwargs):
    return lambda n_cls: net(n_cls=n_cls, **kwargs)


def res18(n_cls=5, input_kernel_size=3, block_channel_starts=64):
    return ResNet(BasicBlock, [2, 2, 2, 2], block_channel_starts, n_cls, input_kernel_size)


def res34(n_cls=5, input_kernel_size=3, block_channel_starts=64):
    block_channel_starts //= 2
    return ResNet(BasicBlock, [3, 4, 6, 3], block_channel_starts, n_cls, input_kernel_size)


def res50(n_cls=5, input_kernel_size=3, block_channel_starts=64):
    return ResNet(Bottleneck, [3, 4, 6, 3], block_channel_starts, n_cls, input_kernel_size)


def res101(n_cls=5, input_kernel_size=3, block_channel_starts=64):
    return ResNet(Bottleneck, [3, 4, 23, 3], block_channel_starts, n_cls, input_kernel_size)


def res152(n_cls=5, input_kernel_size=3, block_channel_starts=64):
    return ResNet(Bottleneck, [3, 8, 36, 3], block_channel_starts, n_cls, input_kernel_size)


class CLSTM(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], n_cls=5):
        super(CLSTM, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.rnn = nn.LSTM(128, 128, 2, batch_first=True, dropout=0)
        self.fc = nn.Linear(128, 5, bias=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        # print('out', out.shape)
        out, (h, c) = self.rnn(out)
        # print('out after rnn', out.shape)
        # |x| = (batch_size, length, hidden_size)
        out = self.fc(out)
        # |x| = (batch_size, length, vocab_size)
        return out


class LSTM(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, output_dim=4, input_dim=12, hidden_dim=32, n_layers=3):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            # dropout = 0.1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x
