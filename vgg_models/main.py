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


def _migrate_classifier_keys(state_dict):
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("classifier.") and not k.startswith("classifier.block."):
            remapped["classifier.block." + k[len("classifier.") :]] = v
        else:
            remapped[k] = v
    return remapped


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
        arr = self.image_tensors[index]
        img = Image.fromarray(arr, mode="L")
        label = self.target_labels[index]
        if self.transform_pipeline is not None:
            img = self.transform_pipeline(img)
        return img, label


parser = argparse.ArgumentParser(description="PyTorch VGG16 QuickDraw Training")
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("-a", "--arch", default="vgg16_quickdraw")
parser.add_argument("--start-epoch", default=0, type=int)
parser.add_argument("-b", "--batch-size", default=128, type=int)
parser.add_argument("--lr", "--learning-rate", default=0.05, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float)
parser.add_argument("--print-freq", "-p", default=100, type=int)
parser.add_argument("--resume", default="", type=str)
parser.add_argument("-e", "--evaluate", action="store_true")
parser.add_argument("--init", type=str, default="")
parser.add_argument("-id", "--device", default="0", type=str)
parser.add_argument("--bit", default=32, type=int)

best_prec = 0
device = torch.device("cpu")
args = parser.parse_args()


def main():
    global args, best_prec, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=> Building VGG16 model...")

    data_root = os.path.join(_ROOT, "dataset")
    class_labels = [f[:-4] for f in sorted(os.listdir(data_root)) if f.endswith(".npy")]
    num_classes = len(class_labels)
    print(f"Number of classes: {num_classes}")

    model = vgg_models.vgg16_quickdraw(num_classes=num_classes, bit=args.bit)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    for param_group in optimizer.param_groups:
        param_group["lr"] = args.lr * (args.batch_size / 128)
    if device.type == "cuda":
        cudnn.benchmark = True

    fdir = os.path.join(_ROOT, "result", f"{args.arch}_{args.bit}bit")
    os.makedirs(fdir, exist_ok=True)

    auto_checkpoint = os.path.join(fdir, "checkpoint.pth")
    auto_best = os.path.join(fdir, "model_best.pth.tar")
    if not args.resume and not args.init:
        if args.evaluate and os.path.isfile(auto_best):
            args.resume = auto_best
        elif os.path.isfile(auto_checkpoint):
            args.resume = auto_checkpoint

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint["epoch"]
            best_prec = checkpoint["best_prec"]
            model.load_state_dict(_migrate_classifier_keys(checkpoint["state_dict"]))
            optimizer.load_state_dict(checkpoint["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print(
                f"=> resumed from epoch {checkpoint['epoch']} (best acc: {best_prec:.2f}%)"
            )
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
            exit()
    elif args.init:
        if os.path.isfile(args.init):
            print("=> loading pre-trained model")
            checkpoint = torch.load(args.init, map_location=device)
            model.load_state_dict(
                _migrate_classifier_keys(checkpoint["state_dict"]), strict=False
            )
        else:
            print("No pre-trained model found!")
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
    eval_transforms = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = QuickDrawSparseDataset(
        data_root, class_labels, 700, 0, training_transforms
    )
    val_dataset = QuickDrawSparseDataset(
        data_root, class_labels, 150, 700, eval_transforms
    )
    test_dataset = QuickDrawSparseDataset(
        data_root, class_labels, 150, 850, eval_transforms
    )

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
        f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}"
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
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
            f"Val Acc: {val_acc:.2f}%  Best Val: {best_prec:.2f}%"
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
    print("Training complete. Evaluating on TEST set...")
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
    losses, top1 = AverageMeter(), AverageMeter()
    batch_time, data_time = AverageMeter(), AverageMeter()
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        data_time.update(time.time() - end)
        input, target = input.to(device, non_blocking=True), target.to(
            device, non_blocking=True
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
            print(
                f"Train [{epoch+1}][{i}/{len(trainloader)}]  "
                f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                f"Acc {top1.val:.2f}% ({top1.avg:.2f}%)"
            )
    return losses.avg, top1.avg


def validate(val_loader, model, criterion, split="Val"):
    losses, top1 = AverageMeter(), AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device, non_blocking=True), target.to(
                device, non_blocking=True
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
                    f"{split}: [{i}/{len(val_loader)}]  "
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
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
