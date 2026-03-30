import torch
import torch.nn as nn
from .quant_layer import QuantReLU


class _BranchesA(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        b = max(in_ch // 4, 8)

        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1),
                nn.Conv2d(in_ch, b, 1, bias=False),
            )
        else:
            self.branch1 = nn.Conv2d(in_ch, b, 1, bias=False)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, b, 1, bias=False),
            nn.Conv2d(b, b, 3, stride=stride, padding=1, bias=False),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, b, 1, bias=False),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.Conv2d(b, b, 3, stride=stride, padding=1, bias=False),
        )

        self.project = nn.Conv2d(3 * b, out_ch, 1, bias=False)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return self.project(torch.cat([b1, b2, b3], dim=1))


class _BranchesB(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        b = max(in_ch // 4, 8)

        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1),
                nn.Conv2d(in_ch, b, 1, bias=False),
            )
        else:
            self.branch1 = nn.Conv2d(in_ch, b, 1, bias=False)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, b, 1, bias=False),
            nn.Conv2d(b, b, (1, 3), padding=(0, 1), bias=False),
            nn.Conv2d(b, b, (3, 1), stride=stride, padding=(1, 0), bias=False),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, b, 1, bias=False),
            nn.Conv2d(b, b, 3, stride=stride, padding=1, bias=False),
        )

        self.project = nn.Conv2d(3 * b, out_ch, 1, bias=False)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return self.project(torch.cat([b1, b2, b3], dim=1))


class _BranchesC(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        b = max(in_ch // 4, 8)

        if stride > 1:
            self.branch1 = nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1),
                nn.Conv2d(in_ch, b, 1, bias=False),
            )
        else:
            self.branch1 = nn.Conv2d(in_ch, b, 1, bias=False)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, b, 1, bias=False),
            nn.Conv2d(b, b, (1, 3), padding=(0, 1), bias=False),
            nn.Conv2d(b, b, (3, 1), stride=stride, padding=(1, 0), bias=False),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, b, 1, bias=False),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.Conv2d(b, b, (1, 3), padding=(0, 1), bias=False),
            nn.Conv2d(b, b, (3, 1), stride=stride, padding=(1, 0), bias=False),
        )

        self.project = nn.Conv2d(3 * b, out_ch, 1, bias=False)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return self.project(torch.cat([b1, b2, b3], dim=1))


class Dummy(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        return self.block(x)


class Oneway(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        return self.relu(self.bn(self.conv(x)))


class Twoways(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None, downsample=None):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.downsample = downsample
        self.idem = False

    def forward(self, x, identity):
        if self.idem:
            return x
        x = self.bn(self.conv(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        return self.relu(x)


class InceptionV4Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        branch_cls,
        expand_ratio=2,
        stride=1,
        norm_layer=None,
        bit=32,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        mid_channels = in_channels * expand_ratio
        has_skip = stride == 1 and in_channels == out_channels

        self.idem = False
        self.inter = False

        self.part1 = Oneway(
            branch_cls(in_channels, mid_channels, stride=stride),
            norm_layer(mid_channels),
            QuantReLU(inplace=True, bit=bit),
        )

        downsample = None
        if not has_skip:

            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm_layer(out_channels),
            )
        self.part2 = Twoways(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            QuantReLU(inplace=True, bit=bit),
            downsample,
        )

    def forward(self, x):
        if self.idem:
            return x
        identity = x
        out = self.part1(x)
        if self.inter:
            return out
        return self.part2(out, identity)


class InceptionNetV4(nn.Module):

    STAGE_CHANNELS = [32, 32, 64, 128]
    STAGE_BLOCKS = [2, 4, 4]
    EXPAND_RATIO = 2

    def __init__(self, num_classes=345, bit=32, in_channels=1, img_size=28):
        super().__init__()
        self.bit = bit
        C = self.STAGE_CHANNELS

        self.layer0 = Dummy(
            nn.Sequential(
                nn.Conv2d(in_channels, C[0], 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(C[0]),
                QuantReLU(inplace=True, bit=bit),
            )
        )

        self.layer1 = self._make_stage(
            _BranchesA, C[0], C[1], self.STAGE_BLOCKS[0], stride=1
        )

        self.layer2 = self._make_stage(
            _BranchesB, C[1], C[2], self.STAGE_BLOCKS[1], stride=2
        )

        self.layer3 = self._make_stage(
            _BranchesC, C[2], C[3], self.STAGE_BLOCKS[2], stride=2
        )

        final_pool_size = img_size // 4
        self.layer4 = Dummy(
            nn.Sequential(
                nn.AvgPool2d(final_pool_size, stride=1),
                nn.Flatten(1),
                nn.Linear(C[3], num_classes),
            )
        )

        self._init_weights()

    def _make_stage(self, branch_cls, in_ch, out_ch, num_blocks, stride):
        layers = [
            InceptionV4Block(
                in_ch, out_ch, branch_cls, self.EXPAND_RATIO, stride, bit=self.bit
            ),
        ]
        for _ in range(1, num_blocks):
            layers.append(
                InceptionV4Block(
                    out_ch, out_ch, branch_cls, self.EXPAND_RATIO, 1, bit=self.bit
                )
            )
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantReLU):
                m.show_params()


class IF(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x

    def extra_repr(self) -> str:
        return "threshold={:.3f}".format(self.act_alpha.item())


class Spiking(nn.Module):
    def __init__(self, block, T, k=1):
        super().__init__()
        self.block = block
        self.T = T
        self.is_first = False
        self.idem = False
        self.sign = False
        self.k = k

    def forward(self, x):
        if self.idem:
            return x

        threshold = self.block[2].act_alpha.data
        membrane = 0.5 * threshold

        if self.is_first:
            x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)

        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)

        for dt in range(self.T):
            membrane = membrane + x[:, dt]
            if dt == 0:
                spike_train = torch.zeros(
                    membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],
                    device=membrane.device,
                )
            spikes = torch.clamp(
                torch.trunc(membrane / threshold), -self.k if self.sign else 0, self.k
            )
            membrane = membrane - spikes * threshold
            spike_train[:, dt] = spikes

        return spike_train * threshold


class Spiking_Oneway(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None, T=0, k=1):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.T = T
        self.idem = False
        self.sign = False
        self.k = k

    def forward(self, x):
        if self.idem:
            return x

        threshold = self.relu.act_alpha.data
        membrane = 0.5 * threshold

        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.conv(x)
        x = self.bn(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)

        for dt in range(self.T):
            membrane = membrane + x[:, dt]
            if dt == 0:
                spike_train = torch.zeros(
                    membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],
                    device=membrane.device,
                )
            spikes = torch.clamp(
                torch.trunc(membrane / threshold), -self.k if self.sign else 0, self.k
            )
            membrane = membrane - spikes * threshold
            spike_train[:, dt] = spikes

        return spike_train * threshold


class Spiking_Twoways(nn.Module):
    def __init__(self, conv=None, bn=None, relu=None, downsample=None, T=0, k=1):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.downsample = downsample
        self.T = T
        self.idem = False
        self.sign = False
        self.k = k

    def forward(self, x, identity):
        if self.idem:
            return x

        threshold = self.relu.act_alpha.data
        membrane = 0.5 * threshold

        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        identity = identity.flatten(0, 1)

        x = self.conv(x)
        x = self.bn(x)

        if self.downsample is not None:
            x += self.downsample(identity)
        else:
            x += identity

        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)

        for dt in range(self.T):
            membrane = membrane + x[:, dt]
            if dt == 0:
                spike_train = torch.zeros(
                    membrane.shape[:1] + torch.Size([self.T]) + membrane.shape[1:],
                    device=membrane.device,
                )
            spikes = torch.clamp(
                torch.trunc(membrane / threshold), -self.k if self.sign else 0, self.k
            )
            membrane = membrane - spikes * threshold
            spike_train[:, dt] = spikes

        return spike_train * threshold


class last_Spiking(nn.Module):
    def __init__(self, block, T):
        super().__init__()
        self.block = block
        self.T = T
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.block(x)
        train_shape.extend(x.shape[1:])
        x = x.reshape(train_shape)
        return x.sum(dim=1).div(self.T)


class Spiking_InceptionV4Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        branch_cls,
        expand_ratio=2,
        stride=1,
        norm_layer=None,
        bit=4,
        k=1,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        mid_channels = in_channels * expand_ratio
        has_skip = stride == 1 and in_channels == out_channels

        self.T = 2**bit - 1
        self.idem = False
        self.inter = False
        self.k = k

        downsample = None
        if not has_skip:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                norm_layer(out_channels),
            )

        self.part1 = Spiking_Oneway(
            branch_cls(in_channels, mid_channels, stride=stride),
            norm_layer(mid_channels),
            IF(),
            self.T,
            k,
        )
        self.part2 = Spiking_Twoways(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            IF(),
            downsample,
            self.T,
            k,
        )

    def forward(self, x):
        if self.idem:
            return x
        identity = x
        out = self.part1(x)
        if self.inter:
            return out
        return self.part2(out, identity)


class S_InceptionNetV4(nn.Module):

    STAGE_CHANNELS = InceptionNetV4.STAGE_CHANNELS
    STAGE_BLOCKS = InceptionNetV4.STAGE_BLOCKS
    EXPAND_RATIO = InceptionNetV4.EXPAND_RATIO

    def __init__(self, num_classes=345, bit=4, in_channels=1, img_size=28, k=1):
        super().__init__()
        self.bit = bit
        self.T = 2**bit - 1
        self.k = k
        C = self.STAGE_CHANNELS

        self.layer0 = Spiking(
            nn.Sequential(
                nn.Conv2d(in_channels, C[0], 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(C[0]),
                IF(),
            ),
            self.T,
            k,
        )
        self.layer0.is_first = True

        self.layer1 = self._make_stage(
            _BranchesA, C[0], C[1], self.STAGE_BLOCKS[0], stride=1
        )
        self.layer2 = self._make_stage(
            _BranchesB, C[1], C[2], self.STAGE_BLOCKS[1], stride=2
        )
        self.layer3 = self._make_stage(
            _BranchesC, C[2], C[3], self.STAGE_BLOCKS[2], stride=2
        )

        final_pool_size = img_size // 4
        self.layer4 = last_Spiking(
            nn.Sequential(
                nn.AvgPool2d(final_pool_size, stride=1),
                nn.Flatten(1),
                nn.Linear(C[3], num_classes),
            ),
            self.T,
        )

    def _make_stage(self, branch_cls, in_ch, out_ch, num_blocks, stride):
        layers = [
            Spiking_InceptionV4Block(
                in_ch,
                out_ch,
                branch_cls,
                self.EXPAND_RATIO,
                stride,
                bit=self.bit,
                k=self.k,
            ),
        ]
        for _ in range(1, num_blocks):
            layers.append(
                Spiking_InceptionV4Block(
                    out_ch,
                    out_ch,
                    branch_cls,
                    self.EXPAND_RATIO,
                    1,
                    bit=self.bit,
                    k=self.k,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, QuantReLU):
                m.show_params()


def inceptionnetv4_quickdraw(spike=False, num_classes=345, **kwargs):
    if spike:
        return S_InceptionNetV4(
            num_classes=num_classes, in_channels=1, img_size=28, **kwargs
        )
    else:
        return InceptionNetV4(
            num_classes=num_classes, in_channels=1, img_size=28, **kwargs
        )
