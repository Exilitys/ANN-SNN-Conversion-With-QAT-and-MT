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

import inceptionnet_models as inceptionnet


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


parser = argparse.ArgumentParser(description="InceptionNetV4 QAT Training — QuickDraw")
parser.add_argument("--epochs", default=300, type=int, metavar="N")
parser.add_argument("-a", "--arch", default="inceptionnetv4_quickdraw", metavar="ARCH")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N")
parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N")
parser.add_argument("--lr", default=0.1, type=float, metavar="LR")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
parser.add_argument("--weight-decay", default=1e-4, type=float, metavar="W")
parser.add_argument("--print-freq", "-p", default=100, type=int, metavar="N")
parser.add_argument("--resume", default="", type=str, metavar="PATH")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true")
parser.add_argument(
    "--init",
    default="",
    type=str,
    help="path to pre-trained FP32 model to initialise from",
)
parser.add_argument("-id", "--device", default="0", type=str, help="GPU id")
parser.add_argument(
    "--bit", default=32, type=int, help="quantisation bitwidth (32 = full precision)"
)

best_prec = 0
device = torch.device("cpu")
args = parser.parse_args()


def main():
    global args, best_prec, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=> Building InceptionNetV4 model...")

    data_root = os.path.join(_ROOT, "dataset")
    class_labels = [
        f.split(".npy")[0] for f in sorted(os.listdir(data_root)) if f.endswith(".npy")
    ]
    num_classes = len(class_labels)
    print(f"Number of classes: {num_classes}")

    model = inceptionnet.inceptionnetv4_quickdraw(num_classes=num_classes, bit=args.bit)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    for pg in optimizer.param_groups:
        pg["lr"] = args.lr * (args.batch_size / 128)

    if device.type == "cuda":
        cudnn.benchmark = True

    fdir = os.path.join(_ROOT, "result", f"{args.arch}_{args.bit}bit")
    os.makedirs(fdir, exist_ok=True)

    auto_checkpoint = os.path.join(fdir, "checkpoint.pth")
    auto_best = os.path.join(fdir, "model_best.pth.tar")
    if not args.resume and not args.init:
        if args.evaluate and os.path.isfile(auto_best):
            print(f"=> auto-loading best model for evaluation: {auto_best}")
            args.resume = auto_best
        elif os.path.isfile(auto_checkpoint):
            print(f"=> auto-resuming from: {auto_checkpoint}")
            args.resume = auto_checkpoint

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            ckpt = torch.load(args.resume, map_location=device)
            args.start_epoch = ckpt["epoch"]
            best_prec = ckpt["best_prec"]
            model.load_state_dict(ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print(f"=> resumed epoch {ckpt['epoch']} (best acc: {best_prec:.2f}%)")
        else:
            print(f"=> no checkpoint at '{args.resume}'")
            exit()
    elif args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            ckpt = torch.load(args.init)
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            print("No pre-trained model found!")
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

    train_dataset = QuickDrawSparseDataset(data_root, class_labels, 700, 0, train_tf)
    val_dataset = QuickDrawSparseDataset(data_root, class_labels, 150, 700, eval_tf)
    test_dataset = QuickDrawSparseDataset(data_root, class_labels, 150, 850, eval_tf)

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
    )
    valloader = DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
    )
    testloader = DataLoader(
        test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
    )

    print(
        f"  Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}"
    )

    if args.evaluate:
        print("\n=> Evaluating on TEST set...")
        validate(testloader, model, criterion, split="Test")
        model.show_params()
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        if epoch % 10 == 1:
            model.show_params()

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch)
        val_acc = validate(valloader, model, criterion)

        print("\n" + "=" * 65)
        print(
            f"Epoch [{epoch+1}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.2f}%  "
            f"Val Acc: {val_acc:.2f}%  "
            f"Best Val: {best_prec:.2f}%"
        )
        print("=" * 65 + "\n")

        is_best = val_acc > best_prec
        best_prec = max(val_acc, best_prec)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_prec": best_prec,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            fdir,
        )

    print("\n" + "=" * 65)
    print("Training complete — evaluating on TEST set...")
    print("=" * 65)
    test_acc = validate(testloader, model, criterion, split="Test")
    print(f"=> Final Test Accuracy: {test_acc:.2f}%")


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


def train(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(trainloader):
        data_time.update(time.time() - end)
        input, target = (
            input.to(device, non_blocking=True),
            target.to(device, non_blocking=True),
        )

        output = model(input)
        loss = criterion(output, target)
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            eta = time.strftime(
                "%H:%M:%S", time.gmtime(batch_time.avg * (len(trainloader) - i))
            )
            print(
                f"Train [{epoch+1}][{i}/{len(trainloader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Acc {top1.val:.2f}% ({top1.avg:.2f}%)\tETA: {eta}"
            )

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, split="Val"):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = (
                input.to(device, non_blocking=True),
                target.to(device, non_blocking=True),
            )
            output = model(input)
            loss = criterion(output, target)
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print(
                    f"{split}: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Acc {top1.val:.2f}% ({top1.avg:.2f}%)"
                )

    print(f" * {split} Acc: {top1.avg:.2f}%")
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
