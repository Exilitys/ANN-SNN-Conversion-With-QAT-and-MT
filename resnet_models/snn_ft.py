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

import torchvision
import torchvision.transforms as transforms
from resnet_models.quant_layer import QuantReLU
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import resnet_models


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

            fold_bn_into_conv(block.part2.conv, block.part2.bn, T=T)

            if block.part2.downsample is not None:
                fold_bn_into_conv(
                    block.part2.downsample[0], block.part2.downsample[1], T=T
                )

    print(f"=> BN folded into SNN conv weights (BN set to identity, bias ÷ {T}).")


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
            block_input_lambda = prev_lambda

            cur_lambda = k * block.part1.relu.act_alpha.data
            with torch.no_grad():
                _ppn_scale(block.part1.conv, block_input_lambda / cur_lambda)
            part1_out_lambda = cur_lambda

            cur_lambda = k * block.part2.relu.act_alpha.data
            with torch.no_grad():
                _ppn_scale(block.part2.conv, part1_out_lambda / cur_lambda)

            if block.part2.downsample is not None:
                with torch.no_grad():
                    _ppn_scale(
                        block.part2.downsample[0], block_input_lambda / cur_lambda
                    )

            prev_lambda = cur_lambda

    print("=> Parallel parameter normalization applied.")


def copy_act_alpha_from_ann_to_snn(ann, snn):
    ann_relu_modules = [
        m for m in ann.modules() if isinstance(m, QuantReLU) and hasattr(m, "act_alpha")
    ]
    snn_if_modules = [m for m in snn.modules() if m.__class__.__name__ == "IF"]

    if len(ann_relu_modules) != len(snn_if_modules):
        raise RuntimeError(
            f"act_alpha copy mismatch: ANN has {len(ann_relu_modules)} QuantReLU, "
            f"SNN has {len(snn_if_modules)} IF neurons."
        )

    for qa, ifn in zip(ann_relu_modules, snn_if_modules):
        ifn.act_alpha.data.copy_(qa.act_alpha.data)

    print(
        f"=> act_alpha copied from ANN ({len(ann_relu_modules)} layers) → SNN IF neurons"
    )


def replace_relu_with_staircase(net, snn, k):

    act_alpha = snn.layer0.block[2].act_alpha
    net.layer0.block[2] = StaircaseActivation(act_alpha, k)

    for seg_name in ["layer1", "layer2", "layer3"]:
        net_seg = getattr(net, seg_name)
        snn_seg = getattr(snn, seg_name)
        for net_block, snn_block in zip(net_seg, snn_seg):
            act_alpha = snn_block.part1.relu.act_alpha
            net_block.part1.relu = StaircaseActivation(act_alpha, k)
            act_alpha = snn_block.part2.relu.act_alpha
            net_block.part2.relu = StaircaseActivation(act_alpha, k)

    print("=> Proxy ANN: QuantReLU replaced with StaircaseActivation.")


def recalibrate_bn_in_snn(snn, trainloader, device, n_batches=50):
    print(
        f"=> Recalibrating BN running stats to SNN spike distribution ({n_batches} batches)..."
    )

    for m in snn.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= n_batches:
                break
            snn(inputs.to(device))

    snn.eval()
    print("   [OK] BN running stats recalibrated to SNN spike distribution.")


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
    print(
        "   [OK] Proxy ANN BN running stats re-calibrated to real-image distribution."
    )


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

        self.image_tensors = []
        self.target_labels = []

        for label_index, category_name in enumerate(self.class_labels):
            file_path = os.path.join(root_directory, f"{category_name}.npy")
            try:
                category_data = np.load(file_path)[
                    offset : offset + samples_per_category
                ]
                category_data = category_data.reshape(-1, 28, 28).astype(np.uint8)
                self.image_tensors.append(category_data)
                self.target_labels.extend([label_index] * len(category_data))
            except IOError as e:
                print(f"Error loading {file_path}: {e}")

        self.image_tensors = np.concatenate(self.image_tensors, axis=0)
        self.target_labels = np.array(self.target_labels, dtype=np.int64)

    def __len__(self):
        return len(self.target_labels)

    def __getitem__(self, index):
        array_representation = self.image_tensors[index]
        pil_image = Image.fromarray(array_representation, mode="L")
        ground_truth = self.target_labels[index]
        if self.transform_pipeline is not None:
            pil_image = self.transform_pipeline(pil_image)
        return pil_image, ground_truth


parser = argparse.ArgumentParser(description="PyTorch QuickDraw SNN Training")
parser.add_argument(
    "--epochs", default=300, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet18_quickdraw")
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128),only used for train",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=100,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "-ct",
    "--cifar-type",
    default="10",
    type=int,
    metavar="CT",
    help="10 for cifar10,100 for cifar100 (default: 10)",
)
parser.add_argument(
    "--init",
    help="initialize form pre-trained floating point model",
    type=str,
    default="",
)
parser.add_argument("-id", "--device", default="0", type=str, help="gpu device")
parser.add_argument(
    "--bit", default=2, type=int, help="the bit-width of the quantized network"
)
parser.add_argument(
    "-k",
    "--k",
    default=3,
    type=int,
    help="max spike level for SMT-IF (k=1 is original signed IF)",
)
parser.add_argument(
    "-n",
    "--num_epochs",
    default=1,
    type=int,
    help="number of epochs per layer for Phase 1 fine-tuning",
)
parser.add_argument(
    "--e2e-epochs",
    default=10,
    type=int,
    help="number of epochs for Phase 2 end-to-end fine-tuning (default 10)",
)
parser.add_argument(
    "--force", action="store_true", help="Force tuner to always update weights"
)
parser.add_argument(
    "--start-layer",
    default=0,
    type=int,
    metavar="L",
    help="resume fine-tuning from this layer_id (0-based); "
    "loads the per-layer checkpoint saved after layer L-1",
)

best_prec = 0
args = parser.parse_args()


def main():
    global args, best_prec
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    print("=> Building model...")

    data_root = os.path.join(_ROOT, "dataset")
    class_labels = []
    for class_name in sorted(os.listdir(data_root)):
        if class_name.endswith(".npy"):
            class_labels.append(class_name.split(".npy")[0])
    num_classes = len(class_labels)
    print(f"Number of classes: {num_classes}")

    model = resnet_models.resnet18_quickdraw(num_classes=num_classes, bit=args.bit)
    net = resnet_models.resnet18_quickdraw(num_classes=num_classes, bit=args.bit)
    snn = resnet_models.resnet18_quickdraw(
        spike=True, num_classes=num_classes, bit=args.bit, k=args.k
    )
    criterion = nn.CrossEntropyLoss()

    model = model.to(args.device)
    net = net.to(args.device)
    snn = snn.to(args.device)
    if args.device.type == "cuda":
        cudnn.benchmark = True

    fdir = os.path.join(
        _ROOT, "result", str(args.arch) + "_" + str(args.bit) + "bit_ft"
    )
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    if not args.init:
        args.init = os.path.join(
            _ROOT,
            "result",
            str(args.arch) + "_" + str(args.bit) + "bit",
            "model_best.pth.tar",
        )

    if args.init:
        if os.path.isfile(args.init):
            print(f'\n{"="*60}')
            print(f"=> Loading best ANN checkpoint from: {args.init}")
            checkpoint = torch.load(args.init, map_location="cpu")
            saved_acc = checkpoint.get("best_prec", "N/A")
            saved_epoch = checkpoint.get("epoch", "N/A")
            print(f"   Checkpoint epoch : {saved_epoch}")
            print(
                f"   Best ANN val acc : {saved_acc:.2f}%"
                if isinstance(saved_acc, float)
                else f"   Best ANN val acc : {saved_acc}"
            )
            print(f'{"="*60}')

            print("\n=> Copying ANN weights to SNN, net (proxy ANN)...")
            model.load_state_dict(checkpoint["state_dict"])

            snn.load_state_dict(checkpoint["state_dict"], strict=False)
            net.load_state_dict(checkpoint["state_dict"])
            print("   [OK] Conv/BN weights loaded")

            copy_act_alpha_from_ann_to_snn(model, snn)

            print(f"\n=> Building proxy ANN with StaircaseActivation (k={args.k})...")
            replace_relu_with_staircase(net, snn, k=args.k)

            print(f"\n=> ANN → SNN conversion complete.")
            print(f"   bit={args.bit}  k={args.k}  T={2**args.bit - 1} timesteps")
            print(f'{"="*60}\n')
        else:
            print(f"\n[ERROR] No pre-trained ANN model found at: {args.init}")
            print(f"        Run main.py first to train the ANN.")
            exit()

    print("=> loading QuickDraw data...")
    normalize = transforms.Normalize(mean=[0.1678], std=[0.3272])

    training_transforms = transforms.Compose(
        [
            transforms.RandomAffine(degrees=15, translate=(0, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    validation_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = QuickDrawSparseDataset(
        data_root,
        class_labels,
        samples_per_category=700,
        offset=0,
        transform_pipeline=training_transforms,
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    test_dataset = QuickDrawSparseDataset(
        data_root,
        class_labels,
        samples_per_category=150,
        offset=850,
        transform_pipeline=validation_transforms,
    )
    testloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    if args.evaluate:
        ft_model = os.path.join(fdir, "model_best.pth.tar")
        print(f'\n{"="*60}')
        if os.path.isfile(ft_model):
            print(f"=> Loading fine-tuned SNN weights from: {ft_model}")
            ft_ckpt = torch.load(ft_model, map_location="cpu")
            snn.load_state_dict(ft_ckpt["state_dict"])
            print(f"   [OK] Fine-tuned SNN weights loaded")
        else:
            print(f"[WARN] No fine-tuned model found at: {ft_model}")
            print(f"       Evaluating raw ANN→SNN direct conversion (no fine-tuning)")

        recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
        print(f"=> Evaluating SNN on test set ({len(testloader.dataset)} samples)...")
        print(f"   arch={args.arch}  bit={args.bit}  k={args.k}  T={2**args.bit - 1}")
        print(f'{"="*60}')
        validate(testloader, snn, criterion)
        return

    duration = 2**args.bit - 1
    num_layers = 20
    num_blocks = (num_layers - 2) // 6

    model.eval()
    snn.eval()

    net.eval()

    best_acc = -1
    acc = -1

    layer_ckpt = os.path.join(fdir, "layer_checkpoint.pth")

    if args.start_layer == 0 and os.path.isfile(layer_ckpt):
        lc = torch.load(layer_ckpt, map_location="cpu")
        detected_layer = lc.get("layer_id", -1) + 1
        print(f'\n{"="*60}')
        print(
            f"=> Found existing layer_checkpoint.pth (completed layer_id={detected_layer-1})"
        )
        print(f"   Auto-resuming from layer {detected_layer}.")
        print(f"   To start fresh, delete: {layer_ckpt}")
        print(f'{"="*60}\n')
        args.start_layer = detected_layer

    if args.start_layer > 0:
        if os.path.isfile(layer_ckpt):
            print(f"=> Resuming from layer {args.start_layer}, loading {layer_ckpt}")
            lc = torch.load(layer_ckpt, map_location="cpu")
            snn.load_state_dict(lc["state_dict"])
            best_acc = lc.get("best_acc", -1)
            print(f"   Restored best_acc={best_acc:.2f}%")

            net.load_state_dict(lc["state_dict"], strict=False)
            print(f"   Proxy ANN (net) re-synced from checkpoint")

            recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
        else:
            print(
                f"[WARN] --start-layer={args.start_layer} but no layer_checkpoint.pth found in {fdir}. Starting from scratch."
            )
            args.start_layer = 0

    if args.start_layer == 0:
        recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
        best_acc = validate(testloader, snn, nn.CrossEntropyLoss())
        print(f"=> Initial SNN accuracy (before fine-tuning): {best_acc:.2f}%")

    bypass_blocks(model, num_blocks)
    model.layer4.idem = True

    bypass_blocks(snn, num_blocks)
    snn.layer4.idem = True

    if args.start_layer > 0:
        completed_block_pos = args.start_layer // 2
        for pos in range(completed_block_pos):
            seg_id = pos // num_blocks + 1
            blk_id = pos % num_blocks
            getattr(getattr(snn, f"layer{seg_id}"), str(blk_id)).idem = False
            getattr(getattr(model, f"layer{seg_id}"), str(blk_id)).idem = False
        print(
            f"=> Restored idem=False for {completed_block_pos} completed block "
            f"position(s) (segments before resume point)"
        )

    criterion = nn.MSELoss()

    for layer_id in range(args.start_layer, num_layers - 4):

        segment_id = layer_id // 2 // num_blocks + 1
        block_id = layer_id // 2 % num_blocks
        is_odd = layer_id % 2
        print(
            "=======We are tuning Layer %d Segment %d Block %d=========="
            % (layer_id, segment_id, block_id)
        )

        m = getattr(model, "layer" + str(segment_id))
        m = getattr(m, str(block_id))
        m.idem = False
        if is_odd:
            m.inter = False
        else:
            m.inter = True

        m = getattr(net, "layer" + str(segment_id))
        m = getattr(m, str(block_id))
        if is_odd:
            tuner = m.part2
        else:
            tuner = m.part1
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

        m = getattr(snn, "layer" + str(segment_id))
        m = getattr(m, str(block_id))
        record = {k: v.cpu().clone() for k, v in m.state_dict().items()}

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

                m = getattr(snn, "layer" + str(segment_id))
                m = getattr(m, str(block_id))
                m.idem = True

                with torch.no_grad():
                    in_maps = snn(input)

                if is_odd:
                    part1 = m.part1
                    with torch.no_grad():

                        mid_maps_spike = part1(in_maps)

                    in_maps_r = in_maps.sum(1).div(duration).detach()
                    mid_maps_r = mid_maps_spike.sum(1).div(duration).detach()

                    output = tuner(mid_maps_r, in_maps_r)
                else:
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
                if is_odd:
                    m.part2.load_state_dict(_tuner_sd, strict=False)
                else:
                    m.part1.load_state_dict(_tuner_sd, strict=False)

                m = getattr(snn, "layer" + str(segment_id))
                m = getattr(m, str(block_id))
                m.idem = False
                m.inter = False

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
                        "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                        "GradNorm {grad_norm:.4f}".format(
                            epoch,
                            i,
                            len(trainloader),
                            batch_time=batch_time,
                            data_time=data_time,
                            loss=losses,
                            grad_norm=grad_norm,
                        )
                    )

            scheduler.step()

            for i in range(layer_id // 2 + 1, (num_layers - 4) // 2):
                switch_on(snn, i, num_blocks)
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
            for i in range(layer_id // 2 + 1, (num_layers - 4) // 2):
                switch_off(snn, i, num_blocks)
            snn.layer4.idem = True

            if acc > layer_best_acc:
                layer_best_acc = acc
                m_snn = getattr(snn, "layer" + str(segment_id))
                m_snn = getattr(m_snn, str(block_id))
                layer_best_record = {
                    k: v.cpu().clone() for k, v in m_snn.state_dict().items()
                }
                print(f"  [Layer {layer_id}] New layer best: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc

        REVERT_TOL = 0.3
        m = getattr(snn, "layer" + str(segment_id))
        m = getattr(m, str(block_id))
        if layer_best_acc >= layer_baseline_acc - REVERT_TOL or args.force:
            m.load_state_dict(layer_best_record)
            print(
                f"=> Layer {layer_id}: accepted best-epoch "
                f"({layer_best_acc:.2f}% vs baseline {layer_baseline_acc:.2f}%)"
            )
        else:
            m.load_state_dict(record)
            print(
                f"=> Layer {layer_id}: REVERTED "
                f"({layer_best_acc:.2f}% < baseline {layer_baseline_acc:.2f}% - {REVERT_TOL}%)"
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
    print(f'\n=> Fine-tuned SNN saved to {os.path.join(fdir, "model_best.pth.tar")}')

    print("\n" + "=" * 60)
    print("=> Final SNN evaluation on test set:")
    print("=" * 60)
    validate(testloader, snn, nn.CrossEntropyLoss())


def switch_on(model, b_id, num_blocks):
    segment_id = b_id // num_blocks + 1
    block_id = b_id % num_blocks
    m = getattr(model, "layer" + str(segment_id))
    m = getattr(m, str(block_id))
    m.idem = False


def switch_off(model, b_id, num_blocks):
    segment_id = b_id // num_blocks + 1
    block_id = b_id % num_blocks
    m = getattr(model, "layer" + str(segment_id))
    m = getattr(m, str(block_id))
    m.idem = True


def bypass_blocks(model, num_blocks):

    for i in range(num_blocks):
        getattr(model.layer1, str(i)).idem = True
        getattr(model.layer2, str(i)).idem = True

    for i in range(num_blocks - 1):
        getattr(model.layer3, str(i)).idem = True


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec {top1.val:.3f}% ({top1.avg:.3f}%)".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                    )
                )

    print(" * Prec {top1.avg:.3f}% ".format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, "checkpoint.pth")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, "model_best.pth.tar"))


def adjust_learning_rate(optimizer, epoch):
    adjust_list = [150, 225]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.1


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    main()
