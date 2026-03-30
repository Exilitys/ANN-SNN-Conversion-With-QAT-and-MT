import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_layer import QuantReLU


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
    def __init__(self, conv, bn, relu):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        return self.relu(self.bn(self.conv(x)))


class VGG16(nn.Module):

    def __init__(self, num_classes=345, bit=32, in_channels=1):
        super().__init__()
        self.bit = bit

        def _conv(ic, oc):
            return nn.Conv2d(ic, oc, 3, padding=1, bias=False)

        def _relu():
            return QuantReLU(inplace=True, bit=bit)

        cfg = [
            (in_channels, 64, False),
            (64, 64, True),
            (64, 128, False),
            (128, 128, True),
            (128, 256, False),
            (256, 256, False),
            (256, 256, True),
            (256, 512, False),
            (512, 512, False),
            (512, 512, False),
            (512, 512, False),
            (512, 512, False),
            (512, 512, False),
        ]

        self.pool_after = []
        for i, (ic, oc, pool) in enumerate(cfg):
            layer = Oneway(_conv(ic, oc), nn.BatchNorm2d(oc), _relu())
            setattr(self, f"conv{i}", layer)
            if pool:
                setattr(self, f"pool{i}", nn.MaxPool2d(kernel_size=2, stride=2))
                self.pool_after.append(i)

        self.classifier = Dummy(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(512, num_classes),
            )
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_conv(self, x, layer_id):
        layer = getattr(self, f"conv{layer_id}")
        was_idem = layer.idem
        x = layer(x)

        if not was_idem and layer_id in self.pool_after:
            x = getattr(self, f"pool{layer_id}")(x)
        return x

    def forward(self, x):
        for i in range(13):
            x = self._forward_conv(x, i)
        x = self.classifier(x)
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

    def extra_repr(self):
        return f"threshold={self.act_alpha.item():.3f}"


class Spiking_Oneway(nn.Module):
    def __init__(self, conv, bn, relu, T, k=1, pool=None):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.T = T
        self.k = k
        self.pool = pool
        self.idem = False
        self.sign = False

    def forward(self, x):
        if self.idem:
            return x
        threshold = self.relu.act_alpha.data
        membrane = 0.5 * threshold

        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.conv(x)
        x = self.bn(x)
        if self.pool is not None:
            x = self.pool(x)
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


class Spiking_First(nn.Module):
    def __init__(self, conv, bn, relu, T, k=1, pool=None):
        super().__init__()
        self.conv = conv
        self.bn = bn
        self.relu = relu
        self.T = T
        self.k = k
        self.pool = pool
        self.idem = False
        self.sign = False

    def forward(self, x):
        if self.idem:
            return x
        threshold = self.relu.act_alpha.data
        membrane = 0.5 * threshold

        x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
        train_shape = [x.shape[0], x.shape[1]]
        x = x.flatten(0, 1)
        x = self.conv(x)
        x = self.bn(x)
        if self.pool is not None:
            x = self.pool(x)
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


class last_Spiking(nn.Module):
    def __init__(self, classifier, T):
        super().__init__()
        self.block = classifier
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


class S_VGG16(nn.Module):
    POOL_AFTER = {1, 3, 6}

    def __init__(self, num_classes=345, bit=2, in_channels=1, k=1):
        super().__init__()
        self.bit = bit
        self.T = 2**bit - 1
        self.k = k

        def _conv(ic, oc):
            return nn.Conv2d(ic, oc, 3, padding=1, bias=False)

        cfg = [
            (in_channels, 64),
            (64, 64),
            (64, 128),
            (128, 128),
            (128, 256),
            (256, 256),
            (256, 256),
            (256, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
            (512, 512),
        ]

        for i, (ic, oc) in enumerate(cfg):
            pool = None
            if i in self.POOL_AFTER:
                pool = nn.MaxPool2d(kernel_size=2, stride=2)
            relu = IF()
            if i == 0:
                layer = Spiking_First(
                    _conv(ic, oc), nn.BatchNorm2d(oc), relu, self.T, k, pool
                )
            else:
                layer = Spiking_Oneway(
                    _conv(ic, oc), nn.BatchNorm2d(oc), relu, self.T, k, pool
                )
            setattr(self, f"conv{i}", layer)

        self.classifier = last_Spiking(
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(1),
                nn.Linear(512, num_classes),
            ),
            self.T,
        )

    def forward(self, x):
        for i in range(13):
            x = getattr(self, f"conv{i}")(x)
        x = self.classifier(x)
        return x

    def show_params(self):
        for m in self.modules():
            if isinstance(m, IF):
                print(f"IF threshold: {m.act_alpha.item():.3f}")


def vgg16_quickdraw(spike=False, num_classes=345, **kwargs):
    if spike:
        return S_VGG16(num_classes=num_classes, in_channels=1, **kwargs)
    else:
        return VGG16(num_classes=num_classes, in_channels=1, **kwargs)
