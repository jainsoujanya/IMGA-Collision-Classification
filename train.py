import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class VGG19Model(nn.Module):
    """Classifier head on top of a pretrained VGG19 encoder."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        feats = self.avg_pool(feats).flatten(1)
        return self.classifier(feats)


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return train and eval transforms."""
    train_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.2)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_tf, eval_tf


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    val_split: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dataloaders for train/val/test splits using ImageFolder."""
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root '{data_root}' not found.")

    train_dir = data_root / "train"
    test_dir = data_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected 'train' and 'test' folders inside the dataset root.")

    train_tf, eval_tf = build_transforms()
    full_train = datasets.ImageFolder(train_dir.as_posix(), transform=train_tf)
    val_len = int(len(full_train) * val_split)
    train_len = len(full_train) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_len, val_len], generator=generator)
    # Validation should not use augmentation
    val_set.dataset = datasets.ImageFolder(train_dir.as_posix(), transform=eval_tf)

    test_set = datasets.ImageFolder(test_dir.as_posix(), transform=eval_tf)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.grad_scaler.GradScaler,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    epoch_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return {
        "loss": epoch_loss / total,
        "accuracy": correct / total * 100,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    epoch_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        epoch_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return {
        "loss": epoch_loss / total,
        "accuracy": correct / total * 100,
    }


def save_metrics(metrics: Dict, output_dir: Path) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / f"training_metrics_{timestamp}.json"
    with metrics_file.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"📈 Metrics saved to {metrics_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate VGG19 collision detector.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"), help="Directory with train/test folders.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("training_runs"), help="Where to store metrics/checkpoints.")
    parser.add_argument("--checkpoint", type=Path, default=Path("best_model.pth"), help="Path to save best model.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    model = VGG19Model(num_classes=len(train_loader.dataset.dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), args.checkpoint)
            print(f"💾 New best model saved to {args.checkpoint} (Val Acc: {best_val_acc:.2f}%)")

    print("✅ Training finished. Loading best checkpoint for testing...")
    if Path(args.checkpoint).exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print("⚠️  Best checkpoint not found; evaluating final weights.")

    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"🧪 Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.2f}%")

    summary = {
        "config": vars(args),
        "history": history,
        "test": test_metrics,
        "best_val_accuracy": best_val_acc,
    }
    save_metrics(summary, args.output_dir)


if __name__ == "__main__":
    main()


