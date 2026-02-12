import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class VGG19Model(nn.Module):
    """Same architecture used during training and in app.py."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        feats = self.avg_pool(feats).flatten(1)
        return self.classifier(feats)


def build_loader(data_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
    if not data_dir.exists():
        raise FileNotFoundError(f"Test directory '{data_dir}' not found.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = datasets.ImageFolder(data_dir.as_posix(), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total * 100,
        "samples": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate best_model.pth on the test split.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/test"), help="Folder with test images (ImageFolder layout).")
    parser.add_argument("--checkpoint", type=Path, default=Path("best_model.pth"), help="Path to trained weights.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    loader = build_loader(args.data_root, args.batch_size, args.num_workers)
    model = VGG19Model(num_classes=len(loader.dataset.classes)).to(device)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint '{args.checkpoint}' not found. Train the model first.")

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    metrics = evaluate(model, loader, device)

    print(
        f"🧪 Test Results | Loss: {metrics['loss']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.2f}% | Samples: {metrics['samples']}"
    )


if __name__ == "__main__":
    main()


