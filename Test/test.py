import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import resnet_models
import vgg_models
import alexnet_models
import efficientnet_models
import inceptionnet_models
import mobilenetv4_models


E_MAC = 4.6e-12
E_AC = 0.9e-12

HISTORY_FILE = os.path.join(_ROOT, "eval_history.jsonl")


class QuickDrawDataset(Dataset):
    def __init__(self, root, class_labels, n, offset=0, transform=None):
        self.transform = transform
        tensors, labels = [], []
        for idx, name in enumerate(class_labels):
            fp = os.path.join(root, f"{name}.npy")
            data = np.load(fp)[offset : offset + n].reshape(-1, 28, 28).astype(np.uint8)
            tensors.append(data)
            labels.extend([idx] * len(data))
        self.images = np.concatenate(tensors, axis=0)
        self.labels = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i], mode="L")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def build_test_loader(data_root, class_labels, batch_size=50):
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1678], std=[0.3272]),
        ]
    )
    ds = QuickDrawDataset(data_root, class_labels, n=150, offset=850, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def build_train_loader(data_root, class_labels, batch_size=128):
    tf = transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1678], std=[0.3272]),
        ]
    )
    ds = QuickDrawDataset(data_root, class_labels, n=700, offset=0, transform=tf)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )


def _remap_layer4_keys(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("layer4.") and not k.startswith("layer4.block."):
            new_sd["layer4.block." + k[len("layer4.") :]] = v
        else:
            new_sd[k] = v
    return new_sd


def _prepare_state_dict(state_dict, arch):
    if "mobilenetv4" in arch.lower():
        return _remap_layer4_keys(state_dict)
    return state_dict


def fold_bn_into_layer(layer, bn, T=1):
    with torch.no_grad():
        std = (bn.running_var + bn.eps).sqrt()
        factor = bn.weight / std
        if isinstance(layer, nn.Linear):
            layer.weight.data.mul_(factor.view(-1, 1))
        else:
            layer.weight.data.mul_(factor.view(-1, 1, 1, 1))
        absorbed_bias = (bn.bias - bn.running_mean * factor) / T
        if layer.bias is None:
            layer.bias = nn.Parameter(absorbed_bias.clone())
        else:
            layer.bias.data.add_(absorbed_bias)
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()


def fold_all_bn_generic(model, T=1):
    folded = 0
    for _, module in model.named_modules():
        children = list(module.named_children())
        for i in range(len(children) - 1):
            _, child_a = children[i]
            _, child_b = children[i + 1]
            is_compute = isinstance(child_a, (nn.Conv2d, nn.Linear))
            is_bn = isinstance(child_b, (nn.BatchNorm2d, nn.BatchNorm1d))
            if is_compute and is_bn:
                fold_bn_into_layer(child_a, child_b, T=T)
                folded += 1
    print(f"=> BN folded: {folded} layer+BN pairs set to identity (T={T}).")
    if folded == 0:
        print("   [WARN] No conv/linear+BN pairs found — check model structure.")


def recalibrate_bn_in_snn(snn, trainloader, device, n_batches=50):
    print(f"=> Recalibrating BN running stats ({n_batches} batches)...")
    for m in snn.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.train()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= n_batches:
                break
            snn(inputs.to(device))
    snn.eval()
    print("   [OK] BN running stats recalibrated.")


def _use_bn_recalibration(arch):
    arch_lower = arch.lower()
    return any(
        x in arch_lower
        for x in ("resnet", "alexnet", "mobilenet", "vgg", "efficientnet", "inception")
    )


def _get_builder(arch):
    if arch == "vgg16_quickdraw":
        return (
            lambda **kw: vgg_models.vgg16_quickdraw(**kw),
            lambda **kw: vgg_models.vgg16_quickdraw(spike=True, **kw),
        )
    if arch == "resnet18_quickdraw":
        return (
            lambda **kw: resnet_models.resnet18_quickdraw(**kw),
            lambda **kw: resnet_models.resnet18_quickdraw(spike=True, **kw),
        )
    if arch == "alexnet_quickdraw":
        return (
            lambda **kw: alexnet_models.alexnet_quickdraw(**kw),
            lambda **kw: alexnet_models.alexnet_quickdraw(spike=True, **kw),
        )
    if arch == "efficientnetv2_quickdraw":
        return (
            lambda **kw: efficientnet_models.efficientnetv2_quickdraw(**kw),
            lambda **kw: efficientnet_models.efficientnetv2_quickdraw(spike=True, **kw),
        )
    if arch == "inceptionnetv4_quickdraw":
        return (
            lambda **kw: inceptionnet_models.inceptionnetv4_quickdraw(**kw),
            lambda **kw: inceptionnet_models.inceptionnetv4_quickdraw(spike=True, **kw),
        )
    raise ValueError(
        f"Unknown arch '{arch}'. Supported: vgg16_quickdraw, resnet18_quickdraw, "
        "alexnet_quickdraw, efficientnetv2_quickdraw, inceptionnetv4_quickdraw, "
        "mobilenetv4_quickdraw, mobilenetv4convsmall_quickdraw"
    )


def build_ann(arch, num_classes, bit, device):
    ann_fn, _ = _get_builder(arch)
    return ann_fn(num_classes=num_classes, bit=bit).to(device)


def build_snn(arch, num_classes, bit, k, device):
    _, snn_fn = _get_builder(arch)
    return snn_fn(num_classes=num_classes, bit=bit, k=k).to(device)


def copy_act_alpha(ann, snn):
    ann_qa = [
        m
        for m in ann.modules()
        if type(m).__name__ == "QuantReLU" and hasattr(m, "act_alpha")
    ]
    snn_if = [m for m in snn.modules() if type(m).__name__ == "IF"]
    if len(ann_qa) != len(snn_if):
        print(
            f"[WARN] copy_act_alpha: ANN {len(ann_qa)} QuantReLU vs "
            f"SNN {len(snn_if)} IF — copying min({len(ann_qa)}, {len(snn_if)})"
        )
    for qa, ifn in zip(ann_qa, snn_if):
        ifn.act_alpha.data.copy_(qa.act_alpha.data)
    print(f"=> act_alpha copied: {min(len(ann_qa), len(snn_if))} layers")


def load_ckpt(model, path, arch="", strict=True, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    sd = _prepare_state_dict(sd, arch)
    model.load_state_dict(sd, strict=strict)
    return {k: ckpt[k] for k in ckpt if k != "state_dict"}


@torch.no_grad()
def evaluate_accuracy(model, loader, device, topk=(1, 5)):
    model.eval()
    maxk = max(topk)
    correct = {k: 0 for k in topk}
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total += y.size(0)
        _, pred = out.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct_mask = pred.eq(y.view(1, -1).expand_as(pred))
        for k in topk:
            correct[k] += correct_mask[:k].any(dim=0).sum().item()
    return tuple(100.0 * correct[k] / total for k in topk)


def fmt_energy(joules):
    if joules is None:
        return "N/A"
    pJ = joules * 1e12
    if pJ < 1e3:
        return f"{pJ:.2f} pJ"
    nJ = pJ / 1e3
    if nJ < 1e3:
        return f"{nJ:.2f} nJ"
    uJ = nJ / 1e3
    if uJ < 1e3:
        return f"{uJ:.2f} µJ"
    mJ = uJ / 1e3
    if mJ < 1e3:
        return f"{mJ:.2f} mJ"
    return f"{mJ/1e3:.4f} J"


def count_ann_ops(model, in_channels, img_size, device):
    hooks, total_ops = [], [0]

    def _conv_hook(module, inp, out):
        c_in = module.in_channels
        kh, kw = module.kernel_size
        c_out = module.out_channels
        h_out, w_out = out.shape[-2], out.shape[-1]
        total_ops[0] += (c_in // module.groups) * kh * kw * c_out * h_out * w_out

    def _linear_hook(module, inp, out):
        total_ops[0] += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    dummy = torch.zeros(1, in_channels, img_size, img_size, device=device)
    model.eval()
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    return total_ops[0]


class FiringRateMonitor:

    _SPIKING_CLASSES = frozenset(
        {
            "Spiking",
            "Spiking_First",
            "Spiking_Oneway",
            "Spiking_Twoways",
            "Spiking_AlexOneway",
            "Spiking_Linear",
            "last_Spiking",
            "Spiking_first",
            "Spiking_UIBOneway",
            "Spiking_UIBTwoways",
        }
    )

    _FIRST_CLASSES = frozenset(
        {
            "Spiking_First",
            "Spiking_first",
        }
    )

    @staticmethod
    def _is_first_layer(module):
        cname = type(module).__name__
        if cname in FiringRateMonitor._FIRST_CLASSES:
            return True
        if cname == "Spiking" and getattr(module, "is_first", False):
            return True
        return False

    def __init__(self, k):
        self.k = k
        self._layer_rhos = {}
        self._layer_counts = {}
        self._layer_macs = {}
        self._hooks = []

    @staticmethod
    def _threshold(module):
        cname = type(module).__name__

        if cname == "last_Spiking":
            return None

        if hasattr(module, "relu") and hasattr(module.relu, "act_alpha"):
            return module.relu.act_alpha.data.item()

        if type(module).__name__ == "Spiking_first":
            try:
                return module.block[2].act_alpha.data.item()
            except (IndexError, AttributeError):
                pass

        if hasattr(module, "block"):
            for child in module.block:
                if hasattr(child, "act_alpha"):
                    return child.act_alpha.data.item()

        return None

    @staticmethod
    def _get_compute_layer(module):
        cname = type(module).__name__

        if cname == "Spiking_UIBOneway":

            if hasattr(module, "expand_conv") and isinstance(
                module.expand_conv, nn.Conv2d
            ):
                return module.expand_conv

        if cname == "Spiking_UIBTwoways":
            if hasattr(module, "proj_conv") and isinstance(module.proj_conv, nn.Conv2d):
                return module.proj_conv

        if hasattr(module, "linear") and isinstance(module.linear, nn.Linear):
            return module.linear

        if hasattr(module, "conv"):
            c = module.conv
            if isinstance(c, (nn.Conv2d, nn.Linear)):
                return c

            if hasattr(c, "project") and isinstance(c.project, nn.Conv2d):
                return c.project

        if hasattr(module, "block"):
            if isinstance(module.block, nn.Sequential):
                for child in module.block:
                    if isinstance(child, nn.Conv2d):
                        return child
            elif isinstance(module.block, (nn.Conv2d, nn.Linear)):
                return module.block

        return None

    def register(self, model):
        for name, mod in model.named_modules():
            cname = type(mod).__name__
            if cname not in self._SPIKING_CLASSES:
                continue

            layer_key = name

            compute = self._get_compute_layer(mod)
            if compute is not None:

                def _mac_hook(inner_mod, inp, out, key=layer_key):
                    if isinstance(inner_mod, nn.Linear):
                        macs = inner_mod.in_features * inner_mod.out_features
                    else:
                        macs = (
                            (inner_mod.in_channels // inner_mod.groups)
                            * inner_mod.kernel_size[0]
                            * inner_mod.kernel_size[1]
                            * inner_mod.out_channels
                            * out.shape[-2]
                            * out.shape[-1]
                        )
                    self._layer_macs[key] = macs

                self._hooks.append(compute.register_forward_hook(_mac_hook))

            if self._is_first_layer(mod):
                self._layer_rhos[layer_key] = 1.0
                self._layer_counts[layer_key] = 1
                continue

            if cname == "last_Spiking":
                continue

            def _make_hook(key, m):
                def _hook(module, inp, output):
                    theta = self._threshold(module)
                    if theta is None or theta <= 0:
                        return
                    rho = output.detach().abs().mean().item() / (theta * self.k)
                    if key not in self._layer_rhos:
                        self._layer_rhos[key] = rho
                        self._layer_counts[key] = 1
                    else:
                        n = self._layer_counts[key]
                        self._layer_rhos[key] = (self._layer_rhos[key] * n + rho) / (
                            n + 1
                        )
                        self._layer_counts[key] = n + 1

                return _hook

            self._hooks.append(mod.register_forward_hook(_make_hook(layer_key, mod)))

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        self._layer_rhos.clear()
        self._layer_counts.clear()
        self._layer_macs.clear()

    @property
    def global_rho(self):
        if not self._layer_rhos:
            return 0.0
        return float(np.mean(list(self._layer_rhos.values())))

    @property
    def weighted_ops(self):
        if not self._layer_macs:
            return None
        total = 0.0
        for lname, rho in self._layer_rhos.items():
            mac = self._layer_macs.get(lname)
            if mac is None:
                return None
            total += rho * mac
        return total

    @property
    def layer_rhos(self):
        return dict(self._layer_rhos)


@torch.no_grad()
def measure_firing_rate(snn, loader, device, k, max_batches=10):
    monitor = FiringRateMonitor(k)
    monitor.register(snn)
    snn.eval()
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        snn(x.to(device))
    monitor.remove()
    return monitor.global_rho, monitor.layer_rhos, monitor.weighted_ops


def evaluate_one(
    arch,
    bit,
    k,
    T,
    ann_ckpt_path,
    ft_ckpt_path,
    run_label,
    loader,
    trainloader,
    data_root,
    class_labels,
    device,
    model_name="",
    model_version="",
):
    num_classes = len(class_labels)
    in_channels = 1
    img_size = 28
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name or arch,
        "model_version": model_version or "",
        "run_label": run_label,
        "arch": arch,
        "bit": bit,
        "k": k,
        "T": T,
        "ann_ckpt": ann_ckpt_path or "",
        "ft_ckpt": ft_ckpt_path or "",
    }

    print(f'\n{"="*70}')
    print(f"  Evaluating | arch={arch}  bit={bit}  k={k}  T={T}  run={run_label}")
    print(f'{"="*70}')

    ops_ann = None
    ann_top1 = None
    ann_top5 = None
    ann_epochs = None

    if ann_ckpt_path and os.path.isfile(ann_ckpt_path):
        print(f"\n[ANN] Loading {ann_ckpt_path}")
        ann = build_ann(arch, num_classes, bit, device)
        meta = load_ckpt(ann, ann_ckpt_path, arch=arch, strict=True, device=device)
        print(
            f'      epoch={meta.get("epoch","?")}  '
            f'saved_acc={meta.get("best_prec","?")}'
        )
        ann_epochs = meta.get("epoch", None)

        print("[ANN] Measuring accuracy ...")
        ann_top1, ann_top5 = evaluate_accuracy(ann, loader, device)
        print(f"[ANN] Top-1 = {ann_top1:.2f}%   Top-5 = {ann_top5:.2f}%")

        print("[ANN] Counting Ops_ANN ...")
        ops_ann = count_ann_ops(ann, in_channels, img_size, device)
        e_ann = E_MAC * ops_ann
        print(
            f"[ANN] Ops_ANN = {ops_ann:,} MAC/sample  |  "
            f"E_ANN = {fmt_energy(e_ann)}/sample"
        )
    else:
        print("[ANN] Checkpoint not found — skipping ANN evaluation.")

    result["ann_accuracy"] = round(ann_top1, 3) if ann_top1 is not None else None
    result["ann_top5_accuracy"] = round(ann_top5, 3) if ann_top5 is not None else None
    result["ann_epochs"] = ann_epochs
    result["ops_ann"] = ops_ann
    result["e_ann_pJ"] = round(E_MAC * ops_ann * 1e12, 4) if ops_ann else None

    snn_d_top1 = None
    snn_d_top5 = None
    rho_d = None
    rho_d_layers = {}
    ops_snn_d = None
    e_snn_d = None
    e_ratio_d = None

    if ann_ckpt_path and os.path.isfile(ann_ckpt_path):
        print(f"\n[SNN-direct] Building SNN from ANN weights (no fine-tuning) ...")
        ann_for_copy = build_ann(arch, num_classes, bit, device)
        load_ckpt(ann_for_copy, ann_ckpt_path, arch=arch, strict=True, device=device)

        snn_direct = build_snn(arch, num_classes, bit, k, device)
        raw_ckpt = torch.load(ann_ckpt_path, map_location=device)
        raw_sd = _prepare_state_dict(raw_ckpt.get("state_dict", raw_ckpt), arch)
        snn_direct.load_state_dict(raw_sd, strict=False)

        copy_act_alpha(ann_for_copy, snn_direct)

        if _use_bn_recalibration(arch):
            recalibrate_bn_in_snn(snn_direct, trainloader, device, n_batches=50)
        else:
            fold_all_bn_generic(snn_direct, T=T)

        print("[SNN-direct] Measuring accuracy ...")
        snn_d_top1, snn_d_top5 = evaluate_accuracy(snn_direct, loader, device)
        print(f"[SNN-direct] Top-1 = {snn_d_top1:.2f}%   Top-5 = {snn_d_top5:.2f}%")

        print("[SNN-direct] Measuring firing rate ...")
        rho_d, rho_d_layers, rho_d_wops = measure_firing_rate(
            snn_direct, loader, device, k
        )
        print(f"[SNN-direct] Global ρ = {rho_d:.4f}")

        if rho_d_wops is not None:
            ops_snn_d = int(round(T * rho_d_wops))
            print("[SNN-direct] Ops_SNN via exact Σ(ρ_l · MAC_l)")
        elif ops_ann:
            ops_snn_d = int(round(rho_d * T * ops_ann))
            print("[SNN-direct] Ops_SNN via unweighted ρ (fallback)")
        e_snn_d = E_AC * ops_snn_d if ops_snn_d is not None else None
        e_ratio_d = (e_snn_d / (E_MAC * ops_ann)) if (e_snn_d and ops_ann) else None
        if ops_snn_d:
            print(
                f"[SNN-direct] Ops_SNN = {ops_snn_d:,.0f} AC/sample  |  "
                f"E_SNN = {fmt_energy(e_snn_d)}/sample  |  ratio = {e_ratio_d:.4f}"
            )

    result["snn_direct_accuracy"] = (
        round(snn_d_top1, 3) if snn_d_top1 is not None else None
    )
    result["snn_direct_top5_accuracy"] = (
        round(snn_d_top5, 3) if snn_d_top5 is not None else None
    )
    result["firing_rate_direct"] = round(rho_d, 4) if rho_d is not None else None
    result["ops_snn_direct"] = round(ops_snn_d) if ops_snn_d is not None else None
    result["e_snn_direct_pJ"] = (
        round(e_snn_d * 1e12, 4) if e_snn_d is not None else None
    )
    result["e_ratio_direct"] = round(e_ratio_d, 6) if e_ratio_d is not None else None
    result["layer_rhos_direct"] = {k_: round(v, 4) for k_, v in rho_d_layers.items()}

    snn_f_top1 = None
    snn_f_top5 = None
    rho_ft = None
    rho_ft_layers = {}
    ops_snn_ft = None
    e_snn_ft = None
    e_ratio_ft = None
    ft_epochs = None

    if ft_ckpt_path and os.path.isfile(ft_ckpt_path):
        print(f"\n[SNN-ft] Loading {ft_ckpt_path}")
        snn_ft_model = build_snn(arch, num_classes, bit, k, device)
        ft_raw = torch.load(ft_ckpt_path, map_location=device)
        ft_sd = _prepare_state_dict(ft_raw.get("state_dict", ft_raw), arch)
        snn_ft_model.load_state_dict(ft_sd)
        ft_epochs = ft_raw.get("epoch", None)

        if _use_bn_recalibration(arch):
            recalibrate_bn_in_snn(snn_ft_model, trainloader, device, n_batches=50)

        print("[SNN-ft] Measuring accuracy ...")
        snn_f_top1, snn_f_top5 = evaluate_accuracy(snn_ft_model, loader, device)
        print(f"[SNN-ft] Top-1 = {snn_f_top1:.2f}%   Top-5 = {snn_f_top5:.2f}%")

        print("[SNN-ft] Measuring firing rate ...")
        rho_ft, rho_ft_layers, rho_ft_wops = measure_firing_rate(
            snn_ft_model, loader, device, k
        )
        print(f"[SNN-ft] Global ρ = {rho_ft:.4f}")

        if rho_ft_wops is not None:
            ops_snn_ft = int(round(T * rho_ft_wops))
            print("[SNN-ft] Ops_SNN via exact Σ(ρ_l · MAC_l)")
        elif ops_ann:
            ops_snn_ft = int(round(rho_ft * T * ops_ann))
            print("[SNN-ft] Ops_SNN via unweighted ρ (fallback)")
        e_snn_ft = E_AC * ops_snn_ft if ops_snn_ft is not None else None
        e_ratio_ft = (e_snn_ft / (E_MAC * ops_ann)) if (e_snn_ft and ops_ann) else None
        if ops_snn_ft:
            print(
                f"[SNN-ft] Ops_SNN = {ops_snn_ft:,.0f} AC/sample  |  "
                f"E_SNN = {fmt_energy(e_snn_ft)}/sample  |  ratio = {e_ratio_ft:.4f}"
            )
    else:
        print(f"[SNN-ft] Checkpoint not found at {ft_ckpt_path} — skipping.")

    result["snn_ft_accuracy"] = round(snn_f_top1, 3) if snn_f_top1 is not None else None
    result["snn_ft_top5_accuracy"] = (
        round(snn_f_top5, 3) if snn_f_top5 is not None else None
    )
    result["ft_epochs"] = ft_epochs
    result["firing_rate_ft"] = round(rho_ft, 4) if rho_ft is not None else None
    result["ops_snn_ft"] = round(ops_snn_ft) if ops_snn_ft is not None else None
    result["e_snn_ft_pJ"] = round(e_snn_ft * 1e12, 4) if e_snn_ft is not None else None
    result["e_ratio_ft"] = round(e_ratio_ft, 6) if e_ratio_ft is not None else None
    result["layer_rhos_ft"] = {k_: round(v, 4) for k_, v in rho_ft_layers.items()}

    return result


def discover_checkpoints(root):
    found = []
    result_dir = os.path.join(root, "result")
    best_dir = os.path.join(root, "best_result")

    arch_bit_map = {
        "alexnet_quickdraw_2bit": ("alexnet_quickdraw", 2, 3),
        "alexnet_quickdraw_3bit": ("alexnet_quickdraw", 3, 7),
        "alexnet_quickdraw_4bit": ("alexnet_quickdraw", 4, 15),
        "resnet18_quickdraw_2bit": ("resnet18_quickdraw", 2, 3),
        "resnet18_quickdraw_3bit": ("resnet18_quickdraw", 3, 7),
        "resnet18_quickdraw_4bit": ("resnet18_quickdraw", 4, 15),
        "vgg16_quickdraw_2bit": ("vgg16_quickdraw", 2, 3),
        "vgg16_quickdraw_3bit": ("vgg16_quickdraw", 3, 7),
        "vgg16_quickdraw_4bit": ("vgg16_quickdraw", 4, 15),
        "efficientnetv2_quickdraw_2bit": ("efficientnetv2_quickdraw", 2, 3),
        "efficientnetv2_quickdraw_3bit": ("efficientnetv2_quickdraw", 3, 7),
        "efficientnetv2_quickdraw_4bit": ("efficientnetv2_quickdraw", 4, 15),
        "inceptionnetv4_quickdraw_2bit": ("inceptionnetv4_quickdraw", 2, 3),
        "inceptionnetv4_quickdraw_3bit": ("inceptionnetv4_quickdraw", 3, 7),
        "inceptionnetv4_quickdraw_4bit": ("inceptionnetv4_quickdraw", 4, 15),
        "mobilenetv4convsmall_quickdraw_2bit": ("mobilenetv4convsmall_quickdraw", 2, 3),
        "mobilenetv4convsmall_quickdraw_3bit": ("mobilenetv4convsmall_quickdraw", 3, 7),
        "mobilenetv4convsmall_quickdraw_4bit": (
            "mobilenetv4convsmall_quickdraw",
            4,
            15,
        ),
        "mobilenetv4_quickdraw_2bit": ("mobilenetv4_quickdraw", 2, 3),
        "mobilenetv4_quickdraw_3bit": ("mobilenetv4_quickdraw", 3, 7),
        "mobilenetv4_quickdraw_4bit": ("mobilenetv4_quickdraw", 4, 15),
    }

    for key, (arch, bit, k) in arch_bit_map.items():
        T = 2**bit - 1
        ann_dir = os.path.join(result_dir, key)
        ft_dir = os.path.join(result_dir, key + "_ft")
        ann_main = os.path.join(ann_dir, "model_best.pth.tar")
        ft_main = os.path.join(ft_dir, "model_best.pth.tar")
        if os.path.isfile(ann_main) or os.path.isfile(ft_main):
            found.append(
                dict(
                    arch=arch,
                    bit=bit,
                    k=k,
                    T=T,
                    ann_ckpt=ann_main if os.path.isfile(ann_main) else "",
                    ft_ckpt=ft_main if os.path.isfile(ft_main) else "",
                    run_label="result_latest",
                )
            )

        for sub, version in [("", "ann"), ("_ft", "ft")]:
            bdir = os.path.join(best_dir, key + sub)
            if not os.path.isdir(bdir):
                continue
            files = sorted(
                f
                for f in os.listdir(bdir)
                if f.endswith(".pth.tar") or f.endswith(".tar")
            )
            for fname in files:
                fpath = os.path.join(bdir, fname)
                parts = fname.replace(".pth.tar", "").replace(".tar", "").split("-")
                rid = parts[1] if len(parts) >= 2 else "1"
                label = f"best_{version}_run{rid}"
                if version == "ann":
                    existing = next(
                        (
                            e
                            for e in found
                            if e["arch"] == arch
                            and e["bit"] == bit
                            and e["run_label"] == label
                        ),
                        None,
                    )
                    if existing:
                        existing["ann_ckpt"] = fpath
                    else:
                        found.append(
                            dict(
                                arch=arch,
                                bit=bit,
                                k=k,
                                T=T,
                                ann_ckpt=fpath,
                                ft_ckpt="",
                                run_label=label,
                            )
                        )
                else:
                    ann_label = label.replace("ft", "ann")
                    existing = next(
                        (
                            e
                            for e in found
                            if e["arch"] == arch
                            and e["bit"] == bit
                            and e["run_label"] == ann_label
                        ),
                        None,
                    )
                    if existing:
                        existing["ft_ckpt"] = fpath
                    else:
                        found.append(
                            dict(
                                arch=arch,
                                bit=bit,
                                k=k,
                                T=T,
                                ann_ckpt="",
                                ft_ckpt=fpath,
                                run_label=label,
                            )
                        )
    return found


def scan_best_result(root, model_version="1.0"):
    best_dir = os.path.join(root, "best_result")
    if not os.path.isdir(best_dir):
        print(f"[scan] best_result/ not found at {best_dir}")
        return []

    ordered_configs = [
        ("resnet18_quickdraw", 2, 3),
        ("resnet18_quickdraw", 3, 7),
        ("resnet18_quickdraw", 4, 15),
        ("vgg16_quickdraw", 2, 3),
        ("vgg16_quickdraw", 3, 7),
        ("vgg16_quickdraw", 4, 15),
        ("alexnet_quickdraw", 2, 3),
        ("alexnet_quickdraw", 3, 7),
        ("alexnet_quickdraw", 4, 15),
        ("inceptionnetv4_quickdraw", 2, 3),
        ("inceptionnetv4_quickdraw", 3, 7),
        ("inceptionnetv4_quickdraw", 4, 15),
        ("efficientnetv2_quickdraw", 2, 3),
        ("efficientnetv2_quickdraw", 3, 7),
        ("efficientnetv2_quickdraw", 4, 15),
    ]

    def _find_ckpt(folder):
        if not os.path.isdir(folder):
            return ""
        candidates = sorted(
            f
            for f in os.listdir(folder)
            if f.endswith(".pth.tar") or f.endswith(".tar")
        )
        for fname in candidates:
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                return fpath
        return ""

    runs = []
    for run_idx, (arch, bit, k) in enumerate(ordered_configs, start=1):
        T = 2**bit - 1
        key = f"{arch}_{bit}bit"
        ann_folder = os.path.join(best_dir, key)
        ft_folder = os.path.join(best_dir, key + "_ft")
        ann_ckpt = _find_ckpt(ann_folder)
        ft_ckpt = _find_ckpt(ft_folder)

        if not ann_ckpt and not ft_ckpt:
            print(
                f"[scan] run_{run_idx:02d}  {arch} {bit}bit — no checkpoints found, skipping."
            )
            continue

        runs.append(
            dict(
                arch=arch,
                bit=bit,
                k=k,
                T=T,
                ann_ckpt=ann_ckpt,
                ft_ckpt=ft_ckpt,
                run_label=f"run_{run_idx}",
                model_version=model_version,
            )
        )
        print(
            f"[scan] run_{run_idx:02d}  {arch} {bit}bit  "
            f'ANN={os.path.basename(ann_ckpt) or "—"}  '
            f'FT={os.path.basename(ft_ckpt) or "—"}'
        )

    return runs


def append_history(record):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n[history] Result appended to {HISTORY_FILE}")


def load_history():
    if not os.path.isfile(HISTORY_FILE):
        return []
    records = []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def print_history_table():
    records = load_history()
    if not records:
        print("No evaluation history found.")
        return

    def _v(r, key):
        v = r.get(key)
        return "N/A" if v is None else str(v)

    print(f'\n{"="*70}')
    print(f"  EVALUATION HISTORY  ({len(records)} record(s))")
    print(f'{"="*70}')

    for idx, r in enumerate(records):
        mname = r.get("model_name", r.get("arch", ""))
        mver = r.get("model_version", "")
        ver_str = f" v{mver}" if mver else ""
        print(
            f'\n  [{idx+1}] {mname}{ver_str}  |  run={r.get("run_label","")}  '
            f'bit={_v(r,"bit")}  k={_v(r,"k")}  T={_v(r,"T")}  '
            f'@ {r.get("timestamp","")}'
        )
        print(f'  {"─"*66}')
        print(
            f'  {"Metric":<28} {"ANN":>12} {"SNN (direct)":>14} '
            f'{"SNN (fine-tuned)":>16}'
        )
        print(f'  {"─"*28} {"─"*12} {"─"*14} {"─"*16}')

        def _acc(key):
            v = r.get(key)
            return f"{v:.2f}%" if v is not None else "N/A"

        def _ratio(key):
            v = r.get(key)
            if v is None:
                return "N/A"
            return f"{v:.4f} ({(1-v)*100:.1f}% saved)"

        def _ops(key):
            v = r.get(key)
            if v is None:
                return "N/A"
            if v >= 1e9:
                return f"{v/1e9:.3f} G"
            if v >= 1e6:
                return f"{v/1e6:.3f} M"
            return str(v)

        rows = [
            ("Epochs trained", _v(r, "ann_epochs"), "—", _v(r, "ft_epochs")),
            (
                "Top-1 Accuracy",
                _acc("ann_accuracy"),
                _acc("snn_direct_accuracy"),
                _acc("snn_ft_accuracy"),
            ),
            (
                "Top-5 Accuracy",
                _acc("ann_top5_accuracy"),
                _acc("snn_direct_top5_accuracy"),
                _acc("snn_ft_top5_accuracy"),
            ),
            (
                "Ops (MACs / ACs)",
                _ops("ops_ann"),
                _ops("ops_snn_direct"),
                _ops("ops_snn_ft"),
            ),
            (
                "Avg firing rate ρ",
                "1.000",
                _v(r, "firing_rate_direct"),
                _v(r, "firing_rate_ft"),
            ),
            (
                "Energy ratio",
                "1.000 (ref)",
                _ratio("e_ratio_direct"),
                _ratio("e_ratio_ft"),
            ),
        ]
        for label, ann_v, direct_v, ft_v in rows:
            print(f"  {label:<28} {ann_v:>12} {direct_v:>14} {ft_v:>16}")
    print(f'\n{"="*70}')


def print_summary(result):
    print(f'\n{"─"*96}')
    mname = result.get("model_name", result.get("arch", ""))
    mver = result.get("model_version", "")
    print(
        f'  SUMMARY  |  {mname}{" v"+mver if mver else ""}  '
        f'bit={result["bit"]}  k={result.get("k","?")}  '
        f'T={result.get("T","?")}  run={result["run_label"]}'
    )
    print(f'{"─"*96}')

    def _fmt_acc(v):
        return f"{v:.2f}%" if v is not None else "N/A"

    def _fmt_ops(v):
        if v is None:
            return "N/A"
        if v >= 1e9:
            return f"{v/1e9:.3f} G"
        if v >= 1e6:
            return f"{v/1e6:.3f} M"
        return f"{v:.0f}"

    def _fmt_e(pJ):
        return fmt_energy(pJ / 1e12 if pJ else None)

    def _fmt_er(v):
        return f"{v:.4f}" if v is not None else "N/A"

    def _fmt_saving(v):
        return f"{(1-v)*100:.1f}%" if v is not None else "N/A"

    T = result.get("T", "?")
    k = result.get("k", "?")
    rows = [
        ("Metric", "ANN", "SNN (direct)", "SNN (fine-tuned)"),
        ("─" * 20, "─" * 20, "─" * 20, "─" * 20),
        ("Bit / k / T", f'{result["bit"]}b/k={k}/T={T}', "←", "←"),
        (
            "Epochs trained",
            str(result.get("ann_epochs") or "—"),
            "—",
            str(result.get("ft_epochs") or "—"),
        ),
        ("─" * 20, "─" * 20, "─" * 20, "─" * 20),
        (
            "Top-1 Accuracy",
            _fmt_acc(result.get("ann_accuracy")),
            _fmt_acc(result.get("snn_direct_accuracy")),
            _fmt_acc(result.get("snn_ft_accuracy")),
        ),
        (
            "Top-5 Accuracy",
            _fmt_acc(result.get("ann_top5_accuracy")),
            _fmt_acc(result.get("snn_direct_top5_accuracy")),
            _fmt_acc(result.get("snn_ft_top5_accuracy")),
        ),
        (
            "Ops (MACs/ACs)",
            _fmt_ops(result.get("ops_ann")),
            _fmt_ops(result.get("ops_snn_direct")),
            _fmt_ops(result.get("ops_snn_ft")),
        ),
        (
            "Energy/sample",
            _fmt_e(result.get("e_ann_pJ")),
            _fmt_e(result.get("e_snn_direct_pJ")),
            _fmt_e(result.get("e_snn_ft_pJ")),
        ),
        (
            "Avg firing rate ρ",
            "1.000 (full)",
            f'{result.get("firing_rate_direct") or "N/A"}',
            f'{result.get("firing_rate_ft") or "N/A"}',
        ),
        (
            "Energy ratio",
            "1.000 (ref)",
            _fmt_er(result.get("e_ratio_direct")),
            _fmt_er(result.get("e_ratio_ft")),
        ),
        (
            "Energy saving",
            "—",
            _fmt_saving(result.get("e_ratio_direct")),
            _fmt_saving(result.get("e_ratio_ft")),
        ),
    ]
    col_w = [22, 22, 22, 22]
    for row in rows:
        print("  ".join(f"{str(v):<{col_w[i]}}" for i, v in enumerate(row)))
    print(f'{"─"*96}')


def parse_args():
    p = argparse.ArgumentParser(description="ANN/SNN evaluation — all architectures")
    p.add_argument(
        "--arch",
        default="",
        help="e.g. resnet18_quickdraw | vgg16_quickdraw | "
        "alexnet_quickdraw | efficientnetv2_quickdraw | "
        "inceptionnetv4_quickdraw | "
        "mobilenetv4_quickdraw | mobilenetv4convsmall_quickdraw",
    )
    p.add_argument(
        "--bit", default=0, type=int, help="quantisation bitwidth (2, 3, or 4)"
    )
    p.add_argument(
        "--k", default=0, type=int, help="max spike level (default: 2^bit − 1)"
    )
    p.add_argument("--ann-ckpt", default="")
    p.add_argument("--ft-ckpt", default="")
    p.add_argument("--run-label", default="manual")
    p.add_argument("--model-name", default="")
    p.add_argument("--model-version", default="")
    p.add_argument("--show-history", action="store_true")
    p.add_argument("--device", default="")
    p.add_argument(
        "--scan",
        action="store_true",
        help=(
            "Automatically scan best_result/ for all paired ANN + FT checkpoints "
            "and evaluate them in sequence (mirrors the commands in test.txt). "
            "Sequential run labels run_1, run_2, … are assigned in the same "
            "order as test.txt.  --model-version is forwarded to every run."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.show_history:
        print_history_table()
        return

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    data_root = os.path.join(_ROOT, "dataset")
    class_labels = sorted(
        f.replace(".npy", "") for f in os.listdir(data_root) if f.endswith(".npy")
    )
    num_classes = len(class_labels)
    print(f"Dataset: {num_classes} classes from {data_root}")
    loader = build_test_loader(data_root, class_labels)
    trainloader = build_train_loader(data_root, class_labels)

    if args.scan:

        print("\n[--scan] Scanning best_result/ for all paired checkpoints …")
        runs = scan_best_result(_ROOT, model_version=args.model_version or "1.0")
        if not runs:
            print(
                "[--scan] No checkpoints found in best_result/.  "
                "Have you copied the .pth.tar files there?"
            )
            return
        print(f"\n[--scan] {len(runs)} run(s) ready to evaluate:")
        for r in runs:
            print(
                f'  {r["run_label"]:10s}  arch={r["arch"]}  bit={r["bit"]}  '
                f'ann={os.path.basename(r["ann_ckpt"]) or "—"}  '
                f'ft={os.path.basename(r["ft_ckpt"]) or "—"}'
            )
    elif args.arch and args.bit:
        bit = args.bit
        k = args.k if args.k else (2**bit - 1)
        T = 2**bit - 1
        runs = [
            dict(
                arch=args.arch,
                bit=bit,
                k=k,
                T=T,
                ann_ckpt=args.ann_ckpt,
                ft_ckpt=args.ft_ckpt,
                run_label=args.run_label,
                model_name=args.model_name,
                model_version=args.model_version,
            )
        ]
    else:
        runs = discover_checkpoints(_ROOT)
        if not runs:
            print("\n[INFO] No checkpoints found automatically.")
            print("       Use --scan to sweep best_result/, or specify manually:")
            print(
                "         python test.py --arch alexnet_quickdraw --bit 2 "
                "--ann-ckpt PATH --ft-ckpt PATH"
            )
            return
        print(f"\nAuto-discovered {len(runs)} run(s):")
        for r in runs:
            print(
                f'  {r["run_label"]:30s} arch={r["arch"]} bit={r["bit"]} '
                f'ann={os.path.basename(r["ann_ckpt"]) or "—"} '
                f'ft={os.path.basename(r["ft_ckpt"]) or "—"}'
            )

    all_results = []
    for run in runs:
        result = evaluate_one(
            arch=run["arch"],
            bit=run["bit"],
            k=run["k"],
            T=run["T"],
            ann_ckpt_path=run["ann_ckpt"],
            ft_ckpt_path=run["ft_ckpt"],
            run_label=run["run_label"],
            loader=loader,
            trainloader=trainloader,
            data_root=data_root,
            class_labels=class_labels,
            device=device,
            model_name=run.get("model_name", args.model_name),
            model_version=run.get("model_version", args.model_version),
        )
        print_summary(result)
        append_history(result)
        all_results.append(result)

    print(f"\n✓ Evaluation complete. {len(all_results)} run(s) saved to {HISTORY_FILE}")
    print(f"  To view history: python test.py --show-history")


if __name__ == "__main__":
    main()
