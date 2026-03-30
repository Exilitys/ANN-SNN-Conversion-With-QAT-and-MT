import argparse
import os
import sys
import time
import shutil

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import vgg_models
from vgg_models.vgg import IF, Spiking_First, Spiking_Oneway
from vgg_models.quant_layer import QuantReLU


def _migrate_classifier_keys(state_dict):
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("classifier.") and not k.startswith("classifier.block."):
            remapped["classifier.block." + k[len("classifier.") :]] = v
        else:
            remapped[k] = v
    return remapped


class StaircaseActivation(nn.Module):
    def __init__(self, act_alpha, k):
        super().__init__()
        self.act_alpha = act_alpha
        self.k = k

    def forward(self, x):
        theta = self.act_alpha
        quantized = theta * torch.clamp(torch.trunc(x / theta), 0, self.k)
        return x + (quantized - x).detach()


def fold_bn_into_conv(conv, bn, T=1):
    with torch.no_grad():
        std = (bn.running_var + bn.eps).sqrt()
        factor = bn.weight / std
        conv.weight.data.mul_(factor.view(-1, 1, 1, 1))
        absorbed_bias = (bn.bias - bn.running_mean * factor) / T
        if conv.bias is None:
            conv.bias = nn.Parameter(absorbed_bias.clone())
        else:
            conv.bias.data.add_(absorbed_bias)
        bn.running_mean.zero_()
        bn.running_var.fill_(1.0)
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()


def fold_all_bn_in_snn(snn, T=1):
    for i in range(13):
        layer = getattr(snn, f"conv{i}")
        fold_bn_into_conv(layer.conv, layer.bn, T=T)
    print(f"=> BN folded into SNN conv weights (BN set to identity, bias ÷ {T}).")


def _ppn_scale(conv, factor):
    conv.weight.data.mul_(factor)
    if conv.bias is not None:
        conv.bias.data.mul_(factor)


def parallel_param_normalization(snn, k=1):
    prev_lambda = torch.tensor(1.0)
    for i in range(13):
        layer = getattr(snn, f"conv{i}")
        cur_lambda = k * layer.relu.act_alpha.data
        with torch.no_grad():
            _ppn_scale(layer.conv, prev_lambda / cur_lambda)
        prev_lambda = cur_lambda
    print("=> Parallel parameter normalization applied.")


def copy_act_alpha_from_ann_to_snn(ann, snn):
    ann_relus = [
        m for m in ann.modules() if isinstance(m, QuantReLU) and hasattr(m, "act_alpha")
    ]
    snn_ifs = [m for m in snn.modules() if isinstance(m, IF)]
    if len(ann_relus) != len(snn_ifs):
        raise RuntimeError(
            f"act_alpha mismatch: ANN {len(ann_relus)} QuantReLU vs SNN {len(snn_ifs)} IF"
        )
    for qa, ifn in zip(ann_relus, snn_ifs):
        ifn.act_alpha.data.copy_(qa.act_alpha.data)
    print(f"=> act_alpha copied: ANN ({len(ann_relus)} layers) → SNN IF neurons")


def replace_relu_with_staircase(net, snn, k):
    for i in range(13):
        snn_layer = getattr(snn, f"conv{i}")
        net_layer = getattr(net, f"conv{i}")
        net_layer.relu = StaircaseActivation(snn_layer.relu.act_alpha, k)
    print("=> Proxy ANN: QuantReLU replaced with StaircaseActivation.")


def recalibrate_bn_in_snn(snn, trainloader, device, n_batches=50):
    print(f"=> Recalibrating BN running stats ({n_batches} batches)...")
    for m in snn.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= n_batches:
                break
            snn(inputs.to(device))
    snn.eval()
    print("   [OK] BN recalibrated.")


def recalibrate_proxy_bn(net, trainloader, device, n_batches=50):
    print(f"=> Recalibrating proxy ANN BN stats ({n_batches} batches)...")
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= n_batches:
                break
            net(inputs.to(device))
    net.eval()
    print("   [OK] Proxy BN recalibrated.")


class QuickDrawSparseDataset(Dataset):
    def __init__(
        self,
        root_directory,
        class_labels,
        samples_per_category,
        offset=0,
        transform_pipeline=None,
    ):
        self.transform_pipeline = transform_pipeline
        image_tensors, target_labels = [], []
        for label_index, category_name in enumerate(class_labels):
            file_path = os.path.join(root_directory, f"{category_name}.npy")
            try:
                data = np.load(file_path)[offset : offset + samples_per_category]
                data = data.reshape(-1, 28, 28).astype(np.uint8)
                image_tensors.append(data)
                target_labels.extend([label_index] * len(data))
            except IOError as e:
                print(f"Error loading {file_path}: {e}")
        self.image_tensors = np.concatenate(image_tensors, axis=0)
        self.target_labels = np.array(target_labels, dtype=np.int64)

    def __len__(self):
        return len(self.target_labels)

    def __getitem__(self, index):
        img = Image.fromarray(self.image_tensors[index], mode="L")
        label = self.target_labels[index]
        if self.transform_pipeline is not None:
            img = self.transform_pipeline(img)
        return img, label


parser = argparse.ArgumentParser(description="PyTorch VGG16 QuickDraw SNN Fine-tuning")
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("-a", "--arch", default="vgg16_quickdraw")
parser.add_argument("--start-epoch", default=0, type=int)
parser.add_argument("-b", "--batch-size", default=128, type=int)
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float)
parser.add_argument("--print-freq", "-p", default=100, type=int)
parser.add_argument("--resume", default="", type=str)
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--init", type=str, default="")
parser.add_argument("-id", "--device", default="0", type=str)
parser.add_argument("--bit", default=2, type=int)
parser.add_argument("-k", "--k", default=3, type=int)
parser.add_argument("-n", "--num_epochs", default=1, type=int)
parser.add_argument("--e2e-epochs", default=10, type=int)
parser.add_argument("--force", action="store_true")
parser.add_argument("--start-layer", default=0, type=int)

best_prec = 0
args = parser.parse_args()

NUM_CONV = 13


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        correct[:k].reshape(-1).float().sum(0).mul_(100.0 / batch_size) for k in topk
    ]


def validate(val_loader, model, criterion):
    losses, top1, batch_time = AverageMeter(), AverageMeter(), AverageMeter()
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(args.device), target.to(args.device)
            output = model(input)
            loss = criterion(output, target)
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print(
                    f"Test: [{i}/{len(val_loader)}]  "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                    f"Acc {top1.val:.3f}% ({top1.avg:.3f}%)"
                )
    print(f" * Acc {top1.avg:.3f}%")
    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, "checkpoint.pth")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, "model_best.pth.tar"))


def adjust_learning_rate(optimizer, epoch):
    if epoch in [150, 225]:
        for pg in optimizer.param_groups:
            pg["lr"] *= 0.1


def main():
    global args, best_prec
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

    data_root = os.path.join(_ROOT, "dataset")
    class_labels = [f[:-4] for f in sorted(os.listdir(data_root)) if f.endswith(".npy")]
    num_classes = len(class_labels)
    print(f"Number of classes: {num_classes}")

    model = vgg_models.vgg16_quickdraw(num_classes=num_classes, bit=args.bit)
    net = vgg_models.vgg16_quickdraw(num_classes=num_classes, bit=args.bit)
    snn = vgg_models.vgg16_quickdraw(
        spike=True, num_classes=num_classes, bit=args.bit, k=args.k
    )
    criterion = nn.CrossEntropyLoss()

    model = model.to(args.device)
    net = net.to(args.device)
    snn = snn.to(args.device)
    if args.device.type == "cuda":
        cudnn.benchmark = True

    fdir = os.path.join(_ROOT, "result", f"{args.arch}_{args.bit}bit_ft")
    os.makedirs(fdir, exist_ok=True)

    if not args.init:
        args.init = os.path.join(
            _ROOT, "result", f"{args.arch}_{args.bit}bit", "model_best.pth.tar"
        )

    if args.init:
        if os.path.isfile(args.init):
            print(f'\n{"="*60}')
            print(f"=> Loading ANN checkpoint: {args.init}")
            checkpoint = torch.load(args.init, map_location="cpu")
            saved_acc = checkpoint.get("best_prec", "N/A")
            print(
                f"   Best ANN val acc: {saved_acc:.2f}%"
                if isinstance(saved_acc, float)
                else f"   Best ANN val acc: {saved_acc}"
            )
            sd = _migrate_classifier_keys(checkpoint["state_dict"])
            model.load_state_dict(sd)
            snn.load_state_dict(sd, strict=False)
            net.load_state_dict(sd)
            print("   [OK] Weights loaded")
            copy_act_alpha_from_ann_to_snn(model, snn)
            replace_relu_with_staircase(net, snn, k=args.k)
            print(f"   bit={args.bit}  k={args.k}  T={2**args.bit - 1}")
            print(f'{"="*60}\n')
        else:
            print(f"\n[ERROR] No pre-trained ANN found at: {args.init}")
            print("        Run main.py first to train the ANN.")
            exit()

    normalize = transforms.Normalize(mean=[0.1678], std=[0.3272])
    training_transforms = transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = QuickDrawSparseDataset(
        data_root, class_labels, 700, 0, training_transforms
    )
    test_dataset = QuickDrawSparseDataset(
        data_root, class_labels, 150, 850, val_transforms
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    testloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    if args.evaluate:
        ft_model = os.path.join(fdir, "model_best.pth.tar")
        if os.path.isfile(ft_model):
            print(f"=> Loading fine-tuned SNN: {ft_model}")
            ft_ckpt = torch.load(ft_model, map_location="cpu")
            snn.load_state_dict(ft_ckpt["state_dict"])
        else:
            print(
                f"[WARN] No fine-tuned model at {ft_model}; evaluating direct conversion."
            )
        recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
        validate(testloader, snn, criterion)
        return

    duration = 2**args.bit - 1
    model.eval()
    snn.eval()
    net.eval()

    best_acc = -1.0

    layer_ckpt = os.path.join(fdir, "layer_checkpoint.pth")
    if args.start_layer == 0 and os.path.isfile(layer_ckpt):
        lc = torch.load(layer_ckpt, map_location="cpu")
        detected = lc.get("layer_id", -1) + 1
        print(f'\n{"="*60}')
        print(
            f"=> Auto-resuming from layer {detected} (completed layer_id={detected-1})"
        )
        print(f"   To start fresh, delete: {layer_ckpt}")
        print(f'{"="*60}\n')
        args.start_layer = detected

    if args.start_layer > 0:
        if os.path.isfile(layer_ckpt):
            print(f"=> Resuming from layer {args.start_layer}, loading {layer_ckpt}")
            lc = torch.load(layer_ckpt, map_location="cpu")
            snn.load_state_dict(lc["state_dict"])
            best_acc = lc.get("best_acc", -1)
            net.load_state_dict(lc["state_dict"], strict=False)
            recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
            print(f"   best_acc={best_acc:.2f}%")
        else:
            print(
                f"[WARN] --start-layer={args.start_layer} but no checkpoint found. Starting fresh."
            )
            args.start_layer = 0

    if args.start_layer == 0:
        recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
        best_acc = validate(testloader, snn, nn.CrossEntropyLoss())
        print(f"=> Initial SNN accuracy (before fine-tuning): {best_acc:.2f}%")

    for i in range(NUM_CONV):
        getattr(snn, f"conv{i}").idem = i >= args.start_layer
        getattr(model, f"conv{i}").idem = i >= args.start_layer
    snn.classifier.idem = True
    model.classifier.idem = True

    mse_criterion = nn.MSELoss()

    for layer_id in range(args.start_layer, NUM_CONV):
        print(f'\n{"="*60}')
        print(f"=> Tuning layer conv{layer_id}")
        print(f'{"="*60}')

        getattr(model, f"conv{layer_id}").idem = False

        tuner = getattr(net, f"conv{layer_id}")
        tuner.idem = False
        tuner.train()
        tuner.relu.act_alpha.requires_grad_(False)

        effective_lr = 5e-5
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, tuner.parameters()),
            lr=effective_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.num_epochs), eta_min=effective_lr * 0.1
        )

        snn_layer = getattr(snn, f"conv{layer_id}")
        record = {k: v.cpu().clone() for k, v in snn_layer.state_dict().items()}
        layer_baseline_acc = best_acc
        layer_best_acc = -1.0
        layer_best_record = {k: v.clone() for k, v in record.items()}

        for epoch in range(args.num_epochs):
            losses, batch_time, data_time = (
                AverageMeter(),
                AverageMeter(),
                AverageMeter(),
            )
            end = time.time()

            for i, (input, target) in enumerate(trainloader):
                data_time.update(time.time() - end)
                input = input.to(args.device)
                target = target.to(args.device)

                with torch.no_grad():
                    target_map = model(input)

                if layer_id == 0:
                    in_rates = input.detach()
                else:
                    getattr(snn, f"conv{layer_id}").idem = True
                    with torch.no_grad():
                        in_spikes = snn(input)
                    getattr(snn, f"conv{layer_id}").idem = False
                    in_rates = in_spikes.sum(1).div(duration).detach()

                output = _proxy_forward_single(net, layer_id, in_rates)

                _theta_l = tuner.relu.act_alpha.data
                loss = mse_criterion(output / _theta_l, target_map / _theta_l)
                losses.update(loss.item(), input.size(0))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tuner.parameters(), max_norm=1.0)
                optimizer.step()

                _BN_STAT_KEYS = {"running_mean", "running_var", "num_batches_tracked"}
                tuner_sd = {
                    k: v
                    for k, v in tuner.state_dict().items()
                    if not any(s in k for s in _BN_STAT_KEYS)
                }
                snn_layer.load_state_dict(tuner_sd, strict=False)

                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    grad_norm = (
                        sum(
                            p.grad.norm().item() ** 2
                            for p in tuner.parameters()
                            if p.grad is not None
                        )
                        ** 0.5
                    )
                    print(
                        f"Epoch [{epoch}][{i}/{len(trainloader)}]  "
                        f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                        f"GradNorm {grad_norm:.4f}"
                    )

            scheduler.step()

            for j in range(layer_id, NUM_CONV):
                getattr(snn, f"conv{j}").idem = False
            snn.classifier.idem = False

            for m in snn.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.train()
            with torch.no_grad():
                for _j, (_inp, _) in enumerate(trainloader):
                    if _j >= 20:
                        break
                    snn(_inp.to(args.device))
            snn.eval()

            acc = validate(testloader, snn, nn.CrossEntropyLoss())
            print(
                f"  Layer {layer_id} Epoch [{epoch}] val={acc:.2f}%  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

            for j in range(layer_id + 1, NUM_CONV):
                getattr(snn, f"conv{j}").idem = True
            snn.classifier.idem = True

            if acc > layer_best_acc:
                layer_best_acc = acc
                layer_best_record = {
                    k: v.cpu().clone() for k, v in snn_layer.state_dict().items()
                }
                print(f"  [Layer {layer_id}] New layer best: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc

        REVERT_TOL = 0.3
        if layer_best_acc >= layer_baseline_acc - REVERT_TOL or args.force:
            snn_layer.load_state_dict(layer_best_record)
            print(
                f"=> Layer {layer_id}: accepted ({layer_best_acc:.2f}% vs baseline {layer_baseline_acc:.2f}%)"
            )
        else:
            snn_layer.load_state_dict(record)
            print(
                f"=> Layer {layer_id}: REVERTED ({layer_best_acc:.2f}% < {layer_baseline_acc:.2f}% - {REVERT_TOL}%)"
            )

        net.eval()
        recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)

        torch.save(
            {
                "layer_id": layer_id,
                "state_dict": snn.state_dict(),
                "best_acc": best_acc,
            },
            layer_ckpt,
        )
        print(f"=> Layer {layer_id} checkpoint saved.")

    for i in range(NUM_CONV):
        getattr(snn, f"conv{i}").idem = False
    snn.classifier.idem = False
    model.classifier.idem = False

    torch.save(
        {"state_dict": snn.state_dict()}, os.path.join(fdir, "model_best.pth.tar")
    )
    print(f"\n=> Fine-tuned SNN saved.")

    print("\n" + "=" * 60)
    print("=> Final SNN evaluation on test set:")
    print("=" * 60)
    validate(testloader, snn, nn.CrossEntropyLoss())


def _proxy_forward_single(net, layer_id, in_rates):
    if layer_id == 0:

        x = in_rates
    else:
        x = in_rates

    layer = getattr(net, f"conv{layer_id}")
    x = layer(x)

    if layer_id in {1, 3, 6}:
        x = getattr(net, f"pool{layer_id}")(x)
    return x


if __name__ == "__main__":
    main()
