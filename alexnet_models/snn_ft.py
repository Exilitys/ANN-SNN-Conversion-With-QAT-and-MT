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

from alexnet_models.quant_layer import QuantReLU
import alexnet_models

ALEX_BLOCK_MAP = [
    (1, 0),
    (2, 0),
    (3, 0),
    (3, 1),
]
NUM_TUNABLE = len(ALEX_BLOCK_MAP)


class QuickDrawSparseDataset(Dataset):
    def __init__(
        self,
        root_directory,
        class_labels,
        samples_per_category,
        offset=0,
        transform_pipeline=None,
    ):
        self.root_directory = root_directory
        self.class_labels = class_labels
        self.samples_per_category = samples_per_category
        self.offset = offset
        self.transform_pipeline = transform_pipeline

        image_tensors = []
        target_labels = []

        for label_index, category_name in enumerate(class_labels):
            path = os.path.join(root_directory, f"{category_name}.npy")
            try:
                data = np.load(path)[offset : offset + samples_per_category]
                data = data.reshape(-1, 28, 28).astype(np.uint8)
                image_tensors.append(data)
                target_labels.extend([label_index] * len(data))
            except IOError as e:
                print(f"Error loading {path}: {e}")

        self.image_tensors = np.concatenate(image_tensors, axis=0)
        self.target_labels = np.array(target_labels, dtype=np.int64)

    def __len__(self):
        return len(self.target_labels)

    def __getitem__(self, index):
        pil = Image.fromarray(self.image_tensors[index], mode="L")
        lbl = self.target_labels[index]
        if self.transform_pipeline is not None:
            pil = self.transform_pipeline(pil)
        return pil, lbl


class StaircaseActivation(nn.Module):
    def __init__(self, act_alpha, k):
        super(StaircaseActivation, self).__init__()
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

    fold_bn_into_conv(snn.layer0.block[0], snn.layer0.block[1], T=T)

    for seg_name in ["layer1", "layer2", "layer3"]:
        seg = getattr(snn, seg_name)
        for block in seg:

            fold_bn_into_conv(block.part1.conv, block.part1.bn, T=T)

    print(f"=> BN folded into AlexNet SNN conv weights (bias ÷ {T}).")


def _ppn_scale(conv, factor):
    conv.weight.data.mul_(factor)
    if conv.bias is not None:
        conv.bias.data.mul_(factor)


def parallel_param_normalization(snn, k=1):

    prev_lambda = torch.tensor(1.0)
    cur_lambda = k * snn.layer0.block[2].act_alpha.data
    with torch.no_grad():
        _ppn_scale(snn.layer0.block[0], prev_lambda / cur_lambda)
    prev_lambda = cur_lambda

    for seg_name in ["layer1", "layer2", "layer3"]:
        seg = getattr(snn, seg_name)
        for block in seg:
            cur_lambda = k * block.part1.relu.act_alpha.data
            with torch.no_grad():
                _ppn_scale(block.part1.conv, prev_lambda / cur_lambda)
            prev_lambda = cur_lambda

    print("=> Parallel parameter normalization applied.")


def copy_act_alpha_from_ann_to_snn(ann, snn):
    ann_relus = [
        m for m in ann.modules() if isinstance(m, QuantReLU) and hasattr(m, "act_alpha")
    ]
    snn_ifs = [m for m in snn.modules() if m.__class__.__name__ == "IF"]

    if len(ann_relus) != len(snn_ifs):
        raise RuntimeError(
            f"act_alpha copy mismatch: ANN has {len(ann_relus)} QuantReLU, "
            f"SNN has {len(snn_ifs)} IF neurons."
        )

    for qa, ifn in zip(ann_relus, snn_ifs):
        ifn.act_alpha.data.copy_(qa.act_alpha.data)

    print(f"=> act_alpha copied from ANN ({len(ann_relus)} layers) → SNN IF neurons")


def replace_relu_with_staircase(net, snn, k):

    act_alpha = snn.layer0.block[2].act_alpha
    net.layer0.block[2] = StaircaseActivation(act_alpha, k)

    for seg_name in ["layer1", "layer2", "layer3"]:
        net_seg = getattr(net, seg_name)
        snn_seg = getattr(snn, seg_name)
        for net_block, snn_block in zip(net_seg, snn_seg):
            act_alpha = snn_block.part1.relu.act_alpha
            net_block.part1.relu = StaircaseActivation(act_alpha, k)

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
    print("   [OK] BN running stats recalibrated.")


def recalibrate_proxy_bn_on_real_images(net, trainloader, device, n_batches=50):
    print(
        f"=> Recalibrating proxy ANN BN stats on real images ({n_batches} batches)..."
    )
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= n_batches:
                break
            net(inputs.to(device))
    net.eval()
    print("   [OK] Proxy ANN BN stats re-calibrated.")


def bypass_blocks(model):
    for seg_id, blk_id in ALEX_BLOCK_MAP:
        getattr(getattr(model, f"layer{seg_id}"), str(blk_id)).idem = True


def switch_on(model, layer_id):
    seg_id, blk_id = ALEX_BLOCK_MAP[layer_id]
    getattr(getattr(model, f"layer{seg_id}"), str(blk_id)).idem = False


def switch_off(model, layer_id):
    seg_id, blk_id = ALEX_BLOCK_MAP[layer_id]
    getattr(getattr(model, f"layer{seg_id}"), str(blk_id)).idem = True


parser = argparse.ArgumentParser(description="AlexNet SNN fine-tuning on QuickDraw")
parser.add_argument("--epochs", default=300, type=int, metavar="N")
parser.add_argument("-a", "--arch", metavar="ARCH", default="alexnet_quickdraw")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N")
parser.add_argument("--lr", "--learning-rate", default=0.1, type=float, metavar="LR")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, metavar="W")
parser.add_argument("--print-freq", "-p", default=100, type=int, metavar="N")
parser.add_argument("--resume", default="", type=str, metavar="PATH")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true")
parser.add_argument("--init", default="", type=str)
parser.add_argument("-id", "--device", default="0", type=str)
parser.add_argument("--bit", default=2, type=int)
parser.add_argument("-k", "--k", default=3, type=int, help="max spike level for SMT-IF")
parser.add_argument(
    "-n",
    "--num_epochs",
    default=1,
    type=int,
    help="epochs per layer for Phase 1 fine-tuning",
)
parser.add_argument(
    "--e2e-epochs",
    default=10,
    type=int,
    help="epochs for Phase 2 end-to-end fine-tuning",
)
parser.add_argument(
    "--force", action="store_true", help="always accept fine-tuned weights (no revert)"
)
parser.add_argument(
    "--start-layer",
    default=0,
    type=int,
    metavar="L",
    help="resume from this layer_id (0-based); "
    "loads per-layer checkpoint from layer L-1",
)

best_prec = 0
args = parser.parse_args()


def main():
    global args, best_prec
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    print("=> Building AlexNet models (ANN + proxy + SNN)...")

    data_root = os.path.join(_ROOT, "dataset")
    class_labels = [
        f.split(".npy")[0] for f in sorted(os.listdir(data_root)) if f.endswith(".npy")
    ]
    num_classes = len(class_labels)
    print(f"Number of classes: {num_classes}")

    model = alexnet_models.alexnet_quickdraw(num_classes=num_classes, bit=args.bit)
    net = alexnet_models.alexnet_quickdraw(num_classes=num_classes, bit=args.bit)
    snn = alexnet_models.alexnet_quickdraw(
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
            print(f"=> Loading best ANN checkpoint from: {args.init}")
            ckpt = torch.load(args.init, map_location="cpu")
            saved_acc = ckpt.get("best_prec", "N/A")
            saved_ep = ckpt.get("epoch", "N/A")
            print(f"   Checkpoint epoch : {saved_ep}")
            if isinstance(saved_acc, float):
                print(f"   Best ANN val acc : {saved_acc:.2f}%")
            else:
                print(f"   Best ANN val acc : {saved_acc}")
            print(f'{"="*60}')

            model.load_state_dict(ckpt["state_dict"])

            snn.load_state_dict(ckpt["state_dict"], strict=False)
            net.load_state_dict(ckpt["state_dict"])
            print("   [OK] Conv/BN weights loaded")

            copy_act_alpha_from_ann_to_snn(model, snn)

            print(f"\n=> Building proxy ANN with StaircaseActivation (k={args.k})...")
            replace_relu_with_staircase(net, snn, k=args.k)

            print(f"\n=> ANN → SNN conversion complete.")
            print(f"   bit={args.bit}  k={args.k}  T={2**args.bit - 1} timesteps")
            print(f'{"="*60}\n')
        else:
            print(f"\n[ERROR] No ANN model at: {args.init}")
            print(f"        Run alexnet_main.py first.")
            exit()

    print("=> Loading QuickDraw data...")
    normalize = transforms.Normalize(mean=[0.1678], std=[0.3272])

    train_tf = transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = QuickDrawSparseDataset(
        data_root,
        class_labels,
        samples_per_category=700,
        offset=0,
        transform_pipeline=train_tf,
    )
    test_dataset = QuickDrawSparseDataset(
        data_root,
        class_labels,
        samples_per_category=150,
        offset=850,
        transform_pipeline=eval_tf,
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
        ft_path = os.path.join(fdir, "model_best.pth.tar")
        print(f'\n{"="*60}')
        if os.path.isfile(ft_path):
            print(f"=> Loading fine-tuned SNN weights from: {ft_path}")
            ft_ckpt = torch.load(ft_path, map_location="cpu")
            snn.load_state_dict(ft_ckpt["state_dict"])
            print("   [OK] Fine-tuned SNN weights loaded")
        else:
            print(
                f"[WARN] No fine-tuned model at {ft_path}. "
                f"Evaluating raw ANN→SNN direct conversion."
            )
        recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
        print(f"=> Evaluating SNN on test set ({len(testloader.dataset)} samples)...")
        print(
            f"   arch={args.arch}  bit={args.bit}  k={args.k}  " f"T={2**args.bit - 1}"
        )
        print(f'{"="*60}')
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
        detected_layer = lc.get("layer_id", -1) + 1
        print(f'\n{"="*60}')
        print(f"=> Found layer_checkpoint.pth (completed layer_id={detected_layer-1})")
        print(f"   Auto-resuming from layer {detected_layer}.")
        print(f"   To restart fresh, delete: {layer_ckpt}")
        print(f'{"="*60}\n')
        args.start_layer = detected_layer

    if args.start_layer > 0:
        if os.path.isfile(layer_ckpt):
            print(
                f"=> Resuming from layer {args.start_layer}, " f"loading {layer_ckpt}"
            )
            lc = torch.load(layer_ckpt, map_location="cpu")
            snn.load_state_dict(lc["state_dict"])
            best_acc = lc.get("best_acc", -1.0)
            print(f"   Restored best_acc={best_acc:.2f}%")
            net.load_state_dict(lc["state_dict"], strict=False)
            print("   Proxy ANN (net) re-synced from checkpoint")
            recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
        else:
            print(
                f"[WARN] --start-layer={args.start_layer} but no "
                f"layer_checkpoint.pth found. Starting from scratch."
            )
            args.start_layer = 0

    if args.start_layer == 0:
        recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
        best_acc = validate(testloader, snn, nn.CrossEntropyLoss())
        print(f"=> Initial SNN accuracy (before fine-tuning): {best_acc:.2f}%")

    bypass_blocks(model)
    model.layer4.idem = True
    bypass_blocks(snn)
    snn.layer4.idem = True

    if args.start_layer > 0:
        for pos in range(args.start_layer):
            seg_id, blk_id = ALEX_BLOCK_MAP[pos]
            getattr(getattr(snn, f"layer{seg_id}"), str(blk_id)).idem = False
            getattr(getattr(model, f"layer{seg_id}"), str(blk_id)).idem = False
        print(f"=> Restored idem=False for {args.start_layer} completed block(s)")

    criterion = nn.MSELoss()

    for layer_id in range(args.start_layer, NUM_TUNABLE):
        segment_id, block_id = ALEX_BLOCK_MAP[layer_id]

        print("\n" + "=" * 60)
        print(
            f"=> Tuning Layer {layer_id} "
            f"(layer{segment_id}[{block_id}]  –  part1 only)"
        )
        print("=" * 60)

        ref_m = getattr(model, f"layer{segment_id}")
        ref_m = getattr(ref_m, str(block_id))
        ref_m.idem = False
        ref_m.inter = True

        net_m = getattr(net, f"layer{segment_id}")
        net_m = getattr(net_m, str(block_id))
        tuner = net_m.part1
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

        snn_m = getattr(snn, f"layer{segment_id}")
        snn_m = getattr(snn_m, str(block_id))
        record = {k: v.cpu().clone() for k, v in snn_m.state_dict().items()}

        layer_baseline_acc = best_acc
        layer_best_acc = -1.0
        layer_best_record = {k: v.clone() for k, v in record.items()}

        for epoch in range(args.num_epochs):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            end = time.time()

            for i, (input, target) in enumerate(trainloader):
                data_time.update(time.time() - end)
                input = input.to(args.device)
                target = target.to(args.device)

                with torch.no_grad():
                    target_map = model(input)

                snn_m = getattr(snn, f"layer{segment_id}")
                snn_m = getattr(snn_m, str(block_id))
                snn_m.idem = True
                with torch.no_grad():
                    in_maps = snn(input)
                snn_m.idem = False

                in_maps_r = in_maps.sum(1).div(duration).detach()

                output = tuner(in_maps_r)

                _theta_l = tuner.relu.act_alpha.data
                loss = criterion(output / _theta_l, target_map / _theta_l)
                losses.update(loss.item(), input.size(0))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(tuner.parameters(), max_norm=1.0)
                optimizer.step()

                _BN_STAT_KEYS = {"running_mean", "running_var", "num_batches_tracked"}
                _tuner_sd = {
                    k: v
                    for k, v in tuner.state_dict().items()
                    if not any(s in k for s in _BN_STAT_KEYS)
                }
                snn_m = getattr(snn, f"layer{segment_id}")
                snn_m = getattr(snn_m, str(block_id))
                snn_m.part1.load_state_dict(_tuner_sd, strict=False)
                snn_m.idem = False
                snn_m.inter = False

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
                        "Epoch: [{0}][{1}/{2}]\t"
                        "Time {bt.val:.3f} ({bt.avg:.3f})\t"
                        "Data {dt.val:.3f} ({dt.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "GradNorm {gn:.4f}".format(
                            epoch,
                            i,
                            len(trainloader),
                            bt=batch_time,
                            dt=data_time,
                            loss=losses,
                            gn=grad_norm,
                        )
                    )

            scheduler.step()

            for i in range(layer_id + 1, NUM_TUNABLE):
                switch_on(snn, i)
            snn.layer4.idem = False

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
                f"(lr={scheduler.get_last_lr()[0]:.2e})"
            )

            for i in range(layer_id + 1, NUM_TUNABLE):
                switch_off(snn, i)
            snn.layer4.idem = True

            if acc > layer_best_acc:
                layer_best_acc = acc
                snn_m_ref = getattr(snn, f"layer{segment_id}")
                snn_m_ref = getattr(snn_m_ref, str(block_id))
                layer_best_record = {
                    k: v.cpu().clone() for k, v in snn_m_ref.state_dict().items()
                }
                print(f"  [Layer {layer_id}] New layer best: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc

        REVERT_TOL = 0.3
        snn_m = getattr(snn, f"layer{segment_id}")
        snn_m = getattr(snn_m, str(block_id))
        if layer_best_acc >= layer_baseline_acc - REVERT_TOL or args.force:
            snn_m.load_state_dict(layer_best_record)
            print(
                f"=> Layer {layer_id}: accepted best-epoch "
                f"({layer_best_acc:.2f}% vs baseline {layer_baseline_acc:.2f}%)"
            )
        else:
            snn_m.load_state_dict(record)
            print(
                f"=> Layer {layer_id}: REVERTED "
                f"({layer_best_acc:.2f}% < baseline {layer_baseline_acc:.2f}% "
                f"- {REVERT_TOL}%)"
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
        print(f"=> Layer {layer_id} checkpoint saved to {layer_ckpt}")

    snn.layer4.idem = False
    torch.save(
        {"state_dict": snn.state_dict()}, os.path.join(fdir, "model_best.pth.tar")
    )
    print(
        f"\n=> Fine-tuned SNN saved to " f'{os.path.join(fdir, "model_best.pth.tar")}'
    )

    print("\n" + "=" * 60)
    print("=> Final SNN evaluation on test set:")
    print("=" * 60)
    validate(testloader, snn, nn.CrossEntropyLoss())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

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
                    "Test: [{0}/{1}]\t"
                    "Time {bt.val:.3f} ({bt.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec {top1.val:.3f}% ({top1.avg:.3f}%)".format(
                        i, len(val_loader), bt=batch_time, loss=losses, top1=top1
                    )
                )

    print(f" * Prec {top1.avg:.3f}%")
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


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0).mul_(100.0 / batch_size) for k in topk]


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()
