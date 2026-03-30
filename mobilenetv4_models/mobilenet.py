import torch
import torch.nn as nn
from .quant_layer import QuantReLU

_MV4CONV_SMALL_CFG = [
    (0, 3, 1, 32, 32, 32),
    (0, 3, 1, 32, 96, 64),
    (0, 3, 2, 64, 192, 64),
    (3, 3, 1, 64, 192, 64),
    (3, 3, 1, 64, 256, 96),
    (0, 3, 1, 96, 384, 96),
    (0, 3, 1, 96, 384, 96),
    (3, 3, 2, 96, 384, 128),
    (3, 3, 1, 128, 512, 128),
    (0, 3, 1, 128, 512, 128),
    (0, 0, 1, 128, 512, 128),
]


class Dummy(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        return self.block(x)


class UIBOneway(nn.Module):
    def __init__(
        self,
        start_dw_conv,
        start_dw_bn,
        start_dw_act,
        expand_conv,
        expand_bn,
        expand_act,
        mid_dw_conv,
        mid_dw_bn,
        mid_dw_act,
    ):
        super().__init__()
        self.start_dw_conv = start_dw_conv
        self.start_dw_bn = start_dw_bn
        self.start_dw_act = start_dw_act
        self.expand_conv = expand_conv
        self.expand_bn = expand_bn
        self.expand_act = expand_act
        self.mid_dw_conv = mid_dw_conv
        self.mid_dw_bn = mid_dw_bn
        self.mid_dw_act = mid_dw_act

        self.relu = mid_dw_act if mid_dw_act is not None else expand_act
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        if self.start_dw_conv is not None:
            x = self.start_dw_act(self.start_dw_bn(self.start_dw_conv(x)))
        x = self.expand_act(self.expand_bn(self.expand_conv(x)))
        if self.mid_dw_conv is not None:
            x = self.mid_dw_act(self.mid_dw_bn(self.mid_dw_conv(x)))
        return x


class UIBTwoways(nn.Module):
    def __init__(self, proj_conv, proj_bn, relu_act, use_residual=False):
        super().__init__()
        self.proj_conv = proj_conv
        self.proj_bn = proj_bn
        self.relu = relu_act
        self.use_residual = use_residual
        self.idem = False

    def forward(self, x, identity):
        if self.idem:
            return x
        out = self.proj_bn(self.proj_conv(x))
        if self.use_residual:
            out = out + identity
        return self.relu(out)


class UIBBlock(nn.Module):
    def __init__(
        self,
        start_dw_k,
        mid_dw_k,
        stride,
        in_channels,
        expand_size,
        out_channels,
        bit=32,
    ):
        super().__init__()
        self.idem = False
        self.inter = False

        if start_dw_k > 0:
            start_dw_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=start_dw_k,
                stride=stride,
                padding=start_dw_k // 2,
                groups=in_channels,
                bias=False,
            )
            start_dw_bn = nn.BatchNorm2d(in_channels)
            start_dw_act = QuantReLU(inplace=True, bit=bit)
            _expand_stride = 1
        else:
            start_dw_conv = None
            start_dw_bn = None
            start_dw_act = None
            _expand_stride = 1

        expand_conv = nn.Conv2d(in_channels, expand_size, 1, bias=False)
        expand_bn = nn.BatchNorm2d(expand_size)
        expand_act = QuantReLU(inplace=True, bit=bit)

        if mid_dw_k > 0:
            _mid_stride = stride if start_dw_k == 0 else 1
            mid_dw_conv = nn.Conv2d(
                expand_size,
                expand_size,
                kernel_size=mid_dw_k,
                stride=_mid_stride,
                padding=mid_dw_k // 2,
                groups=expand_size,
                bias=False,
            )
            mid_dw_bn = nn.BatchNorm2d(expand_size)
            mid_dw_act = QuantReLU(inplace=True, bit=bit)
        else:
            mid_dw_conv = None
            mid_dw_bn = None
            mid_dw_act = None

        self.part1 = UIBOneway(
            start_dw_conv,
            start_dw_bn,
            start_dw_act,
            expand_conv,
            expand_bn,
            expand_act,
            mid_dw_conv,
            mid_dw_bn,
            mid_dw_act,
        )

        use_residual = stride == 1 and in_channels == out_channels
        proj_conv = nn.Conv2d(expand_size, out_channels, 1, bias=False)
        proj_bn = nn.BatchNorm2d(out_channels)
        proj_act = QuantReLU(inplace=True, bit=bit)
        self.part2 = UIBTwoways(proj_conv, proj_bn, proj_act, use_residual)

    def forward(self, x):
        if self.idem:
            return x
        identity = x
        out = self.part1(x)
        if self.inter:
            return out
        out = self.part2(out, identity)
        return out


class MobileNetV4ConvSmall(nn.Module):
    def __init__(self, num_classes=345, bit=32, in_channels=1):
        super().__init__()
        self.bit = bit

        self.layer0 = Dummy(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.BatchNorm2d(32),
                QuantReLU(inplace=True, bit=bit),
            )
        )

        cfgs = _MV4CONV_SMALL_CFG
        self.layer1 = nn.Sequential(*[UIBBlock(*cfgs[i], bit=bit) for i in range(0, 4)])
        self.layer2 = nn.Sequential(*[UIBBlock(*cfgs[i], bit=bit) for i in range(4, 8)])
        self.layer3 = nn.Sequential(
            *[UIBBlock(*cfgs[i], bit=bit) for i in range(8, 11)]
        )

        self.layer4 = Dummy(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(128, num_classes),
            )
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

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
                print(f"QuantReLU alpha: {m.act_alpha.item():.3f}")


class IF(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x

    def extra_repr(self):
        return f"threshold={self.act_alpha.item():.4f}"


def _snn_integrate(x_charge, threshold, T, k, sign=False):
    membrane = 0.5 * threshold

    spike_train = None
    for dt in range(T):
        membrane = membrane + x_charge[:, dt]
        if dt == 0:
            spike_train = torch.zeros(
                membrane.shape[:1] + torch.Size([T]) + membrane.shape[1:],
                device=membrane.device,
            )
        if sign:
            spikes = torch.clamp(torch.trunc(membrane / threshold), -k, k)
        else:
            spikes = torch.clamp(torch.trunc(membrane / threshold), 0, k)
        membrane = membrane - spikes * threshold
        spike_train[:, dt] = spikes
    return spike_train * threshold


class Spiking_UIBOneway(nn.Module):
    def __init__(
        self,
        start_dw_conv,
        start_dw_bn,
        start_dw_if,
        expand_conv,
        expand_bn,
        expand_if,
        mid_dw_conv,
        mid_dw_bn,
        mid_dw_if,
        T,
        k,
    ):
        super().__init__()
        self.start_dw_conv = start_dw_conv
        self.start_dw_bn = start_dw_bn
        self.start_dw_if = start_dw_if
        self.expand_conv = expand_conv
        self.expand_bn = expand_bn
        self.expand_if = expand_if
        self.mid_dw_conv = mid_dw_conv
        self.mid_dw_bn = mid_dw_bn
        self.mid_dw_if = mid_dw_if
        self.relu = mid_dw_if if mid_dw_if is not None else expand_if
        self.T = T
        self.k = k
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x

        if self.start_dw_if is not None:
            threshold = self.start_dw_if.act_alpha.data
            train_shape = [x.shape[0], x.shape[1]]
            xf = x.flatten(0, 1)
            xf = self.start_dw_bn(self.start_dw_conv(xf))
            train_shape.extend(xf.shape[1:])
            x = _snn_integrate(xf.reshape(train_shape), threshold, self.T, self.k)

        threshold = self.expand_if.act_alpha.data
        train_shape = [x.shape[0], x.shape[1]]
        xf = x.flatten(0, 1)
        xf = self.expand_bn(self.expand_conv(xf))
        train_shape.extend(xf.shape[1:])
        x = _snn_integrate(xf.reshape(train_shape), threshold, self.T, self.k)

        if self.mid_dw_if is not None:
            threshold = self.mid_dw_if.act_alpha.data
            train_shape = [x.shape[0], x.shape[1]]
            xf = x.flatten(0, 1)
            xf = self.mid_dw_bn(self.mid_dw_conv(xf))
            train_shape.extend(xf.shape[1:])
            x = _snn_integrate(xf.reshape(train_shape), threshold, self.T, self.k)

        return x


class Spiking_UIBTwoways(nn.Module):
    def __init__(self, proj_conv, proj_bn, proj_if, use_residual=False, T=0, k=1):
        super().__init__()
        self.proj_conv = proj_conv
        self.proj_bn = proj_bn
        self.relu = proj_if
        self.use_residual = use_residual
        self.T = T
        self.k = k
        self.idem = False

    def forward(self, x, identity):
        if self.idem:
            return x
        threshold = self.relu.act_alpha.data
        train_shape = [x.shape[0], x.shape[1]]
        xf = x.flatten(0, 1)
        identity_f = identity.flatten(0, 1)
        xf = self.proj_conv(xf)
        xf = self.proj_bn(xf)
        if self.use_residual:
            xf = xf + identity_f
        train_shape.extend(xf.shape[1:])
        x_charge = xf.reshape(train_shape)
        return _snn_integrate(x_charge, threshold, self.T, self.k)


class Spiking_UIBBlock(nn.Module):
    def __init__(
        self,
        start_dw_k,
        mid_dw_k,
        stride,
        in_channels,
        expand_size,
        out_channels,
        bit=4,
        k=1,
    ):
        super().__init__()
        T = 2**bit - 1
        self.idem = False
        self.inter = False

        if start_dw_k > 0:
            start_dw_conv = nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=start_dw_k,
                stride=stride,
                padding=start_dw_k // 2,
                groups=in_channels,
                bias=False,
            )
            start_dw_bn = nn.BatchNorm2d(in_channels)
            start_dw_if = IF()
        else:
            start_dw_conv = None
            start_dw_bn = None
            start_dw_if = None

        expand_conv = nn.Conv2d(in_channels, expand_size, 1, bias=False)
        expand_bn = nn.BatchNorm2d(expand_size)
        expand_if = IF()

        if mid_dw_k > 0:
            _mid_stride = stride if start_dw_k == 0 else 1
            mid_dw_conv = nn.Conv2d(
                expand_size,
                expand_size,
                kernel_size=mid_dw_k,
                stride=_mid_stride,
                padding=mid_dw_k // 2,
                groups=expand_size,
                bias=False,
            )
            mid_dw_bn = nn.BatchNorm2d(expand_size)
            mid_dw_if = IF()
        else:
            mid_dw_conv = None
            mid_dw_bn = None
            mid_dw_if = None

        self.part1 = Spiking_UIBOneway(
            start_dw_conv,
            start_dw_bn,
            start_dw_if,
            expand_conv,
            expand_bn,
            expand_if,
            mid_dw_conv,
            mid_dw_bn,
            mid_dw_if,
            T,
            k,
        )

        use_residual = stride == 1 and in_channels == out_channels
        proj_conv = nn.Conv2d(expand_size, out_channels, 1, bias=False)
        proj_bn = nn.BatchNorm2d(out_channels)
        proj_if = IF()
        self.part2 = Spiking_UIBTwoways(proj_conv, proj_bn, proj_if, use_residual, T, k)

    def forward(self, x):
        if self.idem:
            return x
        identity = x
        out = self.part1(x)
        if self.inter:
            return out
        out = self.part2(out, identity)
        return out


class Spiking_first(nn.Module):
    def __init__(self, block, T, k):
        super().__init__()
        self.block = block
        self.T = T
        self.k = k
        self.idem = False

    def forward(self, x):
        if self.idem:
            return x
        threshold = self.block[2].act_alpha.data
        x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
        train_shape = [x.shape[0], x.shape[1]]
        xf = x.flatten(0, 1)
        xf = self.block[0](xf)
        xf = self.block[1](xf)
        train_shape.extend(xf.shape[1:])
        x_charge = xf.reshape(train_shape)
        return _snn_integrate(x_charge, threshold, self.T, self.k)


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
        xf = x.flatten(0, 1)
        xf = self.block(xf)
        train_shape.extend(xf.shape[1:])
        x = xf.reshape(train_shape)
        return x.sum(dim=1).div(self.T)


class S_MobileNetV4ConvSmall(nn.Module):
    def __init__(self, num_classes=345, bit=4, in_channels=1, k=1):
        super().__init__()
        self.bit = bit
        self.T = 2**bit - 1
        self.k = k

        self.layer0 = Spiking_first(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.BatchNorm2d(32),
                IF(),
            ),
            self.T,
            k,
        )

        cfgs = _MV4CONV_SMALL_CFG
        self.layer1 = nn.Sequential(
            *[Spiking_UIBBlock(*cfgs[i], bit=bit, k=k) for i in range(0, 4)]
        )
        self.layer2 = nn.Sequential(
            *[Spiking_UIBBlock(*cfgs[i], bit=bit, k=k) for i in range(4, 8)]
        )
        self.layer3 = nn.Sequential(
            *[Spiking_UIBBlock(*cfgs[i], bit=bit, k=k) for i in range(8, 11)]
        )

        self.layer4 = last_Spiking(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(128, num_classes),
            ),
            self.T,
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
                print(f"IF threshold: {m.act_alpha.item():.3f}")


def mobilenetv4convsmall_quickdraw(spike=False, num_classes=345, **kwargs):
    if spike:
        return S_MobileNetV4ConvSmall(num_classes=num_classes, in_channels=1, **kwargs)
    else:
        return MobileNetV4ConvSmall(num_classes=num_classes, in_channels=1, **kwargs)
