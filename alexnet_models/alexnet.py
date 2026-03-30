import torch
import torch.nn as nn
from .quant_layer import QuantReLU


class Dummy(nn.Module):
    def __init__(self, block):
        super(Dummy, self).__init__()
        self.block = block
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        return self.block(x)


class AlexOneway(nn.Module):
    def __init__(self, pool=None, conv=None, bn=None, relu=None):
        super(AlexOneway, self).__init__()
        self.pool = pool
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        if self.pool is not None:
            x = self.pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AlexBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=False, bit=32):
        super(AlexBlock, self).__init__()
        self.idem = False
        self.inter = False

        pool_layer = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None
        self.part1 = AlexOneway(
            pool=pool_layer,
            conv=nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            bn=nn.BatchNorm2d(out_channels),
            relu=QuantReLU(inplace=True, bit=bit),
        )

    def forward(self, x):
        if self.idem:
            return x
        out = self.part1(x)
        return out


class AlexNet(nn.Module):
    def __init__(self, num_classes=345, bit=32, in_channels=1, img_size=28):
        super(AlexNet, self).__init__()
        self.bit = bit

        self.layer0 = Dummy(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.BatchNorm2d(64),
                QuantReLU(inplace=True, bit=bit),
            )
        )

        self.layer1 = nn.Sequential(
            AlexBlock(64, 192, pool=True, bit=bit),
        )

        self.layer2 = nn.Sequential(
            AlexBlock(192, 384, pool=True, bit=bit),
        )

        self.layer3 = nn.Sequential(
            AlexBlock(384, 256, pool=False, bit=bit),
            AlexBlock(256, 256, pool=False, bit=bit),
        )

        final_spatial = img_size // 4
        self.layer4 = Dummy(
            nn.Sequential(
                nn.AvgPool2d(kernel_size=final_spatial, stride=1),
                nn.Flatten(1),
                nn.Linear(256, num_classes),
            )
        )

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
        super(IF, self).__init__()
        self.act_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x

    def extra_repr(self) -> str:
        return "threshold={:.3f}".format(self.act_alpha)


class Spiking(nn.Module):
    def __init__(self, block, T, k=1):
        super(Spiking, self).__init__()
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

            if self.sign:
                spikes = torch.clamp(torch.trunc(membrane / threshold), -self.k, self.k)
            else:
                spikes = torch.clamp(torch.trunc(membrane / threshold), 0, self.k)
            membrane = membrane - spikes * threshold

            spike_train[:, dt] = spikes

        return spike_train * threshold


class Spiking_AlexOneway(nn.Module):
    def __init__(self, pool=None, conv=None, bn=None, relu=None, T=0, k=1):
        super(Spiking_AlexOneway, self).__init__()
        self.pool = pool
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.idem = False
        self.T = T
        self.sign = False
        self.k = k

    def forward(self, x):
        if self.idem:
            return x

        threshold = self.relu.act_alpha.data
        membrane = 0.5 * threshold

        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        if self.pool is not None:
            x = self.pool(x)
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

            if self.sign:
                spikes = torch.clamp(torch.trunc(membrane / threshold), -self.k, self.k)
            else:
                spikes = torch.clamp(torch.trunc(membrane / threshold), 0, self.k)
            membrane = membrane - spikes * threshold

            spike_train[:, dt] = spikes

        return spike_train * threshold


class Spiking_AlexBlock(nn.Module):

    def __init__(self, in_channels, out_channels, pool=False, bit=32, k=1):
        super(Spiking_AlexBlock, self).__init__()
        self.idem = False
        self.inter = False
        self.T = 2**bit - 1
        self.k = k

        pool_layer = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None
        self.part1 = Spiking_AlexOneway(
            pool=pool_layer,
            conv=nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            bn=nn.BatchNorm2d(out_channels),
            relu=IF(),
            T=self.T,
            k=k,
        )

    def forward(self, x):
        if self.idem:
            return x
        return self.part1(x)


class last_Spiking(nn.Module):

    def __init__(self, block, T):
        super(last_Spiking, self).__init__()
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


class S_AlexNet(nn.Module):

    def __init__(self, num_classes=345, bit=4, in_channels=1, img_size=28, k=1):
        super(S_AlexNet, self).__init__()
        self.bit = bit
        self.T = 2**bit - 1
        self.k = k

        self.layer0 = Spiking(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.BatchNorm2d(64),
                IF(),
            ),
            T=self.T,
            k=k,
        )
        self.layer0.is_first = True

        self.layer1 = nn.Sequential(
            Spiking_AlexBlock(64, 192, pool=True, bit=bit, k=k),
        )

        self.layer2 = nn.Sequential(
            Spiking_AlexBlock(192, 384, pool=True, bit=bit, k=k),
        )

        self.layer3 = nn.Sequential(
            Spiking_AlexBlock(384, 256, pool=False, bit=bit, k=k),
            Spiking_AlexBlock(256, 256, pool=False, bit=bit, k=k),
        )

        final_spatial = img_size // 4
        self.layer4 = last_Spiking(
            nn.Sequential(
                nn.AvgPool2d(kernel_size=final_spatial, stride=1),
                nn.Flatten(1),
                nn.Linear(256, num_classes),
            ),
            T=self.T,
        )

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, IF):
                print("threshold: {:.3f}".format(m.act_alpha.item()))


def alexnet_quickdraw(
    spike=False, num_classes=345, bit=32, k=1, in_channels=1, img_size=28
):

    if spike:
        return S_AlexNet(
            num_classes=num_classes,
            bit=bit,
            in_channels=in_channels,
            img_size=img_size,
            k=k,
        )
    else:
        return AlexNet(
            num_classes=num_classes, bit=bit, in_channels=in_channels, img_size=img_size
        )
