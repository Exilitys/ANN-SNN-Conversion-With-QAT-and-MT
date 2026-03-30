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

from inceptionnet_models.quant_layer import QuantReLU
import inceptionnet_models

SEGMENT_LAYOUT = [2, 4, 4]


_blocks_schedule = [
    (seg_idx + 1, blk_idx)
    for seg_idx, n_blocks in enumerate(SEGMENT_LAYOUT)
    for blk_idx in range(n_blocks)
]
NUM_TUNABLE_LAYERS = 2 * len(_blocks_schedule)


class StaircaseActivation(nn.Module):
    def __init__(self, act_alpha, k):
        super().__init__()
        self.act_alpha = act_alpha
        self.k = k

    def forward(self, x):
        theta = self.act_alpha
        quantized = theta * torch.clamp(torch.trunc(x / theta), 0, self.k)
        return x + (quantized - x).detach()


def copy_act_alpha_from_ann_to_snn(ann, snn):
    ann_relus = [
        m
        for m in ann.modules()
        if type(m).__name__ == "QuantReLU" and hasattr(m, "act_alpha")
    ]
    snn_ifs = [m for m in snn.modules() if m.__class__.__name__ == "IF"]

    if len(ann_relus) != len(snn_ifs):
        raise RuntimeError(
            f"act_alpha copy mismatch: ANN has {len(ann_relus)} QuantReLU, "
            f"SNN has {len(snn_ifs)} IF neurons."
        )

    for qa, ifn in zip(ann_relus, snn_ifs):
        ifn.act_alpha.data.copy_(qa.act_alpha.data)

    print(f"=> act_alpha copied: {len(ann_relus)} ANN QuantReLU → SNN IF neurons")


def replace_relu_with_staircase(net, snn, k):

    act_alpha = snn.layer0.block[2].act_alpha
    net.layer0.block[2] = StaircaseActivation(act_alpha, k)

    for seg_name in ["layer1", "layer2", "layer3"]:
        net_seg = getattr(net, seg_name)
        snn_seg = getattr(snn, seg_name)
        for net_block, snn_block in zip(net_seg, snn_seg):
            net_block.part1.relu = StaircaseActivation(
                snn_block.part1.relu.act_alpha, k
            )
            net_block.part2.relu = StaircaseActivation(
                snn_block.part2.relu.act_alpha, k
            )

    print("=> Proxy ANN: QuantReLU replaced with StaircaseActivation.")


def recalibrate_bn_in_snn(snn, trainloader, device, n_batches=50):
    print(f"=> Recalibrating SNN BN stats ({n_batches} batches)...")
    for m in snn.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(trainloader):
            if i >= n_batches:
                break
            snn(inputs.to(device))
    snn.eval()
    print("   [OK] SNN BN stats recalibrated.")


def recalibrate_proxy_bn_on_real_images(net, trainloader, device, n_batches=50):
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
    print("   [OK] Proxy ANN BN stats recalibrated.")


def _get_block(model, segment_id, block_id):
    return getattr(getattr(model, f"layer{segment_id}"), str(block_id))


def switch_on(model, layer_id):
    seg, blk = _blocks_schedule[layer_id // 2]
    _get_block(model, seg, blk).idem = False


def switch_off(model, layer_id):
    seg, blk = _blocks_schedule[layer_id // 2]
    _get_block(model, seg, blk).idem = True


def bypass_blocks(model, segment_layout=None):
    if segment_layout is None:
        segment_layout = SEGMENT_LAYOUT
    for seg_idx, n_blocks in enumerate(segment_layout):
        seg = getattr(model, f"layer{seg_idx + 1}")
        for blk_idx in range(n_blocks):
            getattr(seg, str(blk_idx)).idem = True


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
                data = np.load(file_path)[offset : offset + samples_per_category]
                data = data.reshape(-1, 28, 28).astype(np.uint8)
                self.image_tensors.append(data)
                self.target_labels.extend([label_index] * len(data))
            except IOError as e:
                print(f"Error loading {file_path}: {e}")

        self.image_tensors = np.concatenate(self.image_tensors, axis=0)
        self.target_labels = np.array(self.target_labels, dtype=np.int64)

    def __len__(self):
        return len(self.target_labels)

    def __getitem__(self, index):
        pil_image = Image.fromarray(self.image_tensors[index], mode="L")
        ground_truth = self.target_labels[index]
        if self.transform_pipeline is not None:
            pil_image = self.transform_pipeline(pil_image)
        return pil_image, ground_truth


parser = argparse.ArgumentParser(
    description="InceptionNetV4 SNN fine-tuning — QuickDraw"
)
parser.add_argument("--epochs", default=300, type=int, metavar="N")
parser.add_argument("-a", "--arch", default="inceptionnetv4_quickdraw", metavar="ARCH")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N")
parser.add_argument("--lr", default=0.1, type=float, metavar="LR")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
parser.add_argument("--weight-decay", default=1e-4, type=float, metavar="W")
parser.add_argument("--print-freq", "-p", default=100, type=int, metavar="N")
parser.add_argument("--resume", default="", type=str, metavar="PATH")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true")
parser.add_argument("--init", default="", type=str)
parser.add_argument("-id", "--device", default="0", type=str)
parser.add_argument(
    "--bit", default=2, type=int, help="QAT bitwidth — sets T = 2^bit − 1"
)
parser.add_argument(
    "-k",
    "--k",
    default=3,
    type=int,
    help="max spike level for SMT-IF (k = 2^bit − 1 recommended)",
)
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
    help="epochs for Phase 2 end-to-end fine-tuning (reserved)",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="force accept weight update even if accuracy drops",
)
parser.add_argument(
    "--start-layer",
    default=0,
    type=int,
    metavar="L",
    help="resume Phase 1 from this layer_id (0-based)",
)

best_prec = 0
args = parser.parse_args()


def main():
    global args, best_prec
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")
    print("=> Building InceptionNetV4 models (ANN, proxy ANN, SNN)...")

    data_root = os.path.join(_ROOT, "dataset")
    class_labels = [
        f.split(".npy")[0] for f in sorted(os.listdir(data_root)) if f.endswith(".npy")
    ]
    num_classes = len(class_labels)
    print(f"Number of classes: {num_classes}")

    model = inceptionnet_models.inceptionnetv4_quickdraw(
        num_classes=num_classes, bit=args.bit
    )
    net = inceptionnet_models.inceptionnetv4_quickdraw(
        num_classes=num_classes, bit=args.bit
    )
    snn = inceptionnet_models.inceptionnetv4_quickdraw(
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

    if args.init and os.path.isfile(args.init):
        print(f'\n{"="*60}')
        print(f"=> Loading ANN checkpoint: {args.init}")
        ckpt = torch.load(args.init, map_location="cpu")
        saved_acc = ckpt.get("best_prec", "N/A")
        saved_ep = ckpt.get("epoch", "N/A")
        print(f"   Checkpoint epoch  : {saved_ep}")
        if isinstance(saved_acc, float):
            print(f"   Best ANN val acc  : {saved_acc:.2f}%")

        print("\n=> Copying ANN weights → SNN and proxy ANN (net)...")
        model.load_state_dict(ckpt["state_dict"])

        snn.load_state_dict(ckpt["state_dict"], strict=False)
        net.load_state_dict(ckpt["state_dict"])
        print("   [OK] Conv/BN weights loaded.")

        copy_act_alpha_from_ann_to_snn(model, snn)

        print(f"\n=> Building proxy ANN with StaircaseActivation (k={args.k})...")
        replace_relu_with_staircase(net, snn, k=args.k)

        print(f"\n=> ANN → SNN conversion complete.")
        print(f"   bit={args.bit}  k={args.k}  T={2**args.bit - 1} timesteps")
        print(f'{"="*60}\n')
    else:
        print(f"\n[ERROR] No ANN checkpoint at: {args.init}")
        print(f"        Run main_inception.py first.")
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
    val_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = QuickDrawSparseDataset(data_root, class_labels, 700, 0, train_tf)
    test_dataset = QuickDrawSparseDataset(data_root, class_labels, 150, 850, val_tf)

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
        print(f'\n{"="*60}')
        if os.path.isfile(ft_model):
            print(f"=> Loading fine-tuned SNN weights: {ft_model}")
            ft_ckpt = torch.load(ft_model, map_location="cpu")
            snn.load_state_dict(ft_ckpt["state_dict"])
            print("   [OK] Fine-tuned SNN weights loaded.")
        else:
            print(f"[WARN] No fine-tuned model at {ft_model}")
            print("       Evaluating raw ANN→SNN direct conversion.")
        recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
        print(f"=> Evaluating SNN  bit={args.bit}  k={args.k}  T={2**args.bit-1}")
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
        detected = lc.get("layer_id", -1) + 1
        print(f'\n{"="*60}')
        print(f"=> Found layer_checkpoint.pth (completed layer_id={detected-1})")
        print(f"   Auto-resuming from layer {detected}.")
        print(f'{"="*60}\n')
        args.start_layer = detected

    if args.start_layer > 0:
        if os.path.isfile(layer_ckpt):
            print(f"=> Resuming from layer {args.start_layer}: {layer_ckpt}")
            lc = torch.load(layer_ckpt, map_location="cpu")
            snn.load_state_dict(lc["state_dict"])
            best_acc = lc.get("best_acc", -1.0)
            print(f"   Restored best_acc={best_acc:.2f}%")

            net.load_state_dict(lc["state_dict"], strict=False)
            print("   Proxy ANN re-synced from checkpoint.")
            recalibrate_bn_in_snn(snn, trainloader, args.device, n_batches=50)
        else:
            print(
                f"[WARN] --start-layer={args.start_layer} but no "
                f"layer_checkpoint.pth in {fdir}. Starting from scratch."
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
        completed_positions = args.start_layer // 2
        for pos in range(completed_positions):
            seg, blk = _blocks_schedule[pos]
            _get_block(snn, seg, blk).idem = False
            _get_block(model, seg, blk).idem = False
        print(
            f"=> Restored idem=False for {completed_positions} completed "
            f"block position(s) before resume point."
        )

    mse_criterion = nn.MSELoss()

    for layer_id in range(args.start_layer, NUM_TUNABLE_LAYERS):

        schedule_idx = layer_id // 2
        segment_id, block_id = _blocks_schedule[schedule_idx]
        is_odd = layer_id % 2

        print(
            f"\n======= Tuning Layer {layer_id} "
            f"(segment={segment_id}, block={block_id}, "
            f'{"part2" if is_odd else "part1"}) ======='
        )

        m_ref = _get_block(model, segment_id, block_id)
        m_ref.idem = False
        m_ref.inter = not is_odd

        m_net = _get_block(net, segment_id, block_id)
        tuner = m_net.part2 if is_odd else m_net.part1
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

        m_snn = _get_block(snn, segment_id, block_id)
        record = {k: v.cpu().clone() for k, v in m_snn.state_dict().items()}
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

                m_snn = _get_block(snn, segment_id, block_id)
                m_snn.idem = True

                with torch.no_grad():
                    in_maps = snn(input)

                if is_odd:

                    m_snn_block = _get_block(snn, segment_id, block_id)
                    with torch.no_grad():
                        mid_maps_spike = m_snn_block.part1(in_maps)
                    in_maps_r = in_maps.sum(1).div(duration).detach()
                    mid_maps_r = mid_maps_spike.sum(1).div(duration).detach()
                    output = tuner(mid_maps_r, in_maps_r)
                else:

                    in_maps_r = in_maps.sum(1).div(duration).detach()
                    output = tuner(in_maps_r)

                _theta_l = tuner.relu.act_alpha.data
                loss = mse_criterion(output / _theta_l, target_map / _theta_l)
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
                m_snn = _get_block(snn, segment_id, block_id)
                if is_odd:
                    m_snn.part2.load_state_dict(_tuner_sd, strict=False)
                else:
                    m_snn.part1.load_state_dict(_tuner_sd, strict=False)

                m_snn.idem = False
                m_snn.inter = False

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
                        f"Epoch [{epoch}][{i}/{len(trainloader)}]\t"
                        f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                        f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                        f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                        f"GradNorm {grad_norm:.4f}"
                    )

            scheduler.step()

            for i in range(layer_id // 2 + 1, len(_blocks_schedule)):
                seg, blk = _blocks_schedule[i]
                _get_block(snn, seg, blk).idem = False
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

            for i in range(layer_id // 2 + 1, len(_blocks_schedule)):
                seg, blk = _blocks_schedule[i]
                _get_block(snn, seg, blk).idem = True
            snn.layer4.idem = True

            if acc > layer_best_acc:
                layer_best_acc = acc
                m_snn = _get_block(snn, segment_id, block_id)
                layer_best_record = {
                    k: v.cpu().clone() for k, v in m_snn.state_dict().items()
                }
                print(f"  [Layer {layer_id}] New layer best: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc

        REVERT_TOL = 0.3
        m_snn = _get_block(snn, segment_id, block_id)
        if layer_best_acc >= layer_baseline_acc - REVERT_TOL or args.force:
            m_snn.load_state_dict(layer_best_record)
            print(
                f"=> Layer {layer_id}: accepted best-epoch "
                f"({layer_best_acc:.2f}% vs baseline {layer_baseline_acc:.2f}%)"
            )
        else:
            m_snn.load_state_dict(record)
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
        print(f"=> Layer {layer_id} checkpoint saved: {layer_ckpt}")

    snn.layer4.idem = False

    out_path = os.path.join(fdir, "model_best.pth.tar")
    torch.save({"state_dict": snn.state_dict()}, out_path)
    print(f"\n=> Fine-tuned SNN saved to {out_path}")

    print("\n" + "=" * 60)
    print("=> Final SNN evaluation on test set:")
    print("=" * 60)
    validate(testloader, snn, nn.CrossEntropyLoss())


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
                    f"Test: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Prec {top1.val:.3f}% ({top1.avg:.3f}%)"
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
    return [
        correct[:k].reshape(-1).float().sum(0).mul_(100.0 / batch_size) for k in topk
    ]


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main()
