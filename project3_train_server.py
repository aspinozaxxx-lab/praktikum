import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Compose, Normalize, PILToTensor, Resize, ToTensor
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2 import RandomHorizontalFlip, RandomRotation, RandomVerticalFlip


class TransformDataset(Dataset):
    def __init__(self, dataset, transforms):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transforms(x), y


class CachedResizedImageFolder(Dataset):
    def __init__(self, root):
        self.dataset = ImageFolder(root)
        self.classes = self.dataset.classes
        self.resize_to_tensor = Compose([Resize((224, 224)), PILToTensor()])
        self.images = []
        self.labels = []

        for path, label in self.dataset.samples:
            image = self.dataset.loader(path)
            self.images.append(self.resize_to_tensor(image))
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def build_gpu_transforms():
    train_gpu_transforms = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.RandomHorizontalFlip(p=0.2),
            v2.RandomVerticalFlip(p=0.2),
            v2.RandomRotation([-5, 5], fill=1.0),
            v2.Normalize((0.5,), (0.5,)),
        ]
    )
    val_gpu_transforms = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5,), (0.5,)),
        ]
    )
    return train_gpu_transforms, val_gpu_transforms


def build_loaders(data_root, batch_size, num_workers, device, cache_resized):
    if cache_resized:
        print("Caching resized 224x224 tensors in RAM...", flush=True)
        train_dataset = CachedResizedImageFolder(data_root / "train")
        val_dataset = CachedResizedImageFolder(data_root / "test")
    else:
        train_base = ImageFolder(data_root / "train")
        val_base = ImageFolder(data_root / "test")

        test_transforms = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
                Normalize((0.5), (0.5)),
            ]
        )

        train_transforms = Compose(
            [
                Resize((224, 224)),
                RandomHorizontalFlip(p=0.2),
                RandomVerticalFlip(p=0.2),
                RandomRotation([-5, 5], fill=255.0),
                ToTensor(),
                Normalize((0.5), (0.5)),
            ]
        )

        train_dataset = TransformDataset(train_base, train_transforms)
        val_dataset = TransformDataset(val_base, test_transforms)

    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    print("Train images:", len(train_dataset), flush=True)
    print("Val images:", len(val_dataset), flush=True)
    classes = train_dataset.classes if cache_resized else train_dataset.dataset.classes
    print("Classes:", len(classes), flush=True)
    return train_loader, val_loader


def build_model(num_classes, train_mode):
    model = mobilenet_v3_small(weights="IMAGENET1K_V1")
    model.classifier = nn.Linear(in_features=576, out_features=num_classes, bias=True)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    if train_mode == "last_stage":
        for param in model.features[-1].parameters():
            param.requires_grad = True
    elif train_mode == "all":
        for param in model.parameters():
            param.requires_grad = True

    return model


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch_index,
    max_batches,
    profile,
    gpu_transforms,
):
    running_loss = 0.0
    last_loss = 0.0
    total_loss = 0.0
    batch_count = 0
    end = time.perf_counter()
    data_times = []
    gpu_times = []

    for batch_index, data in enumerate(train_loader):
        data_ready = time.perf_counter()
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if gpu_transforms is not None:
            inputs = gpu_transforms(inputs)

        if epoch_index == 0 and batch_index == 0:
            print("First batch inputs device:", inputs.device, flush=True)
            print("First batch labels device:", labels.device, flush=True)

        if profile and device.type == "cuda":
            torch.cuda.synchronize()
            gpu_start = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(inputs)

        if epoch_index == 0 and batch_index == 0:
            print("First batch outputs device:", outputs.device, flush=True)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if profile and device.type == "cuda":
            torch.cuda.synchronize()
            gpu_end = time.perf_counter()
            data_times.append(data_ready - end)
            gpu_times.append(gpu_end - gpu_start)
            end = time.perf_counter()

        running_loss += loss.item()
        total_loss += loss.item()
        batch_count += 1
        if batch_index % 20 == 19:
            last_loss = running_loss / 20.0
            print(f"Epoch: {epoch_index}, batch: {batch_index}, loss {last_loss}", flush=True)
            running_loss = 0.0

        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    if profile and data_times and gpu_times:
        avg_data = sum(data_times) / len(data_times) * 1000
        avg_gpu = sum(gpu_times) / len(gpu_times) * 1000
        print(f"Profile avg data wait ms/batch: {avg_data:.1f}", flush=True)
        print(f"Profile avg GPU step ms/batch: {avg_gpu:.1f}", flush=True)

    return total_loss / max(batch_count, 1)


def validate(model, val_loader, criterion, device, gpu_transforms):
    running_vloss = 0.0

    with torch.no_grad():
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device, non_blocking=True)
            vlabels = vlabels.to(device, non_blocking=True)
            if gpu_transforms is not None:
                vinputs = gpu_transforms(vinputs)
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss

    return running_vloss / (i + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/opt/prak/data/ogyeiv2/ogyeiv2")
    parser.add_argument("--output-dir", default="/opt/prak/project3_runs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--cache-resized", action="store_true")
    parser.add_argument(
        "--train-mode",
        choices=["classifier", "last_stage", "all"],
        default="classifier",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, training would run on CPU.")

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    print("Device:", device, flush=True)
    print("CUDA device:", torch.cuda.get_device_name(0), flush=True)

    train_loader, val_loader = build_loaders(
        data_root,
        args.batch_size,
        args.num_workers,
        device,
        args.cache_resized,
    )
    train_gpu_transforms, val_gpu_transforms = (
        build_gpu_transforms() if args.cache_resized else (None, None)
    )
    num_classes = (
        len(train_loader.dataset.classes)
        if args.cache_resized
        else len(train_loader.dataset.dataset.classes)
    )

    model = build_model(num_classes, args.train_mode)
    model = model.to(device)
    print("Model parameter device:", next(model.parameters()).device, flush=True)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Train mode:", args.train_mode, flush=True)
    print("Trainable params:", trainable_params, flush=True)
    print("Total params:", total_params, flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    best_vloss = 1e5

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}", flush=True)
        model.train(True)
        avg_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            args.max_train_batches,
            args.profile,
            train_gpu_transforms,
        )

        model.eval()
        avg_vloss = validate(model, val_loader, criterion, device, val_gpu_transforms)

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = output_dir / f"color_classifier_{epoch}.pt"
            torch.save(model.state_dict(), model_path)
            print("Saved:", model_path, flush=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
            allocated_mb = torch.cuda.memory_allocated() / 1024**2
            max_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"CUDA memory allocated MB: {allocated_mb:.1f}", flush=True)
            print(f"CUDA max memory allocated MB: {max_allocated_mb:.1f}", flush=True)

        print(f"End epoch train loss {avg_loss}, val loss {avg_vloss.item()}", flush=True)


if __name__ == "__main__":
    main()
