from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from ..model import IMAGE_SIZE, build_classifier

DEFAULT_DATA_ROOT = Path("datasets")
DEFAULT_OUTPUT_ROOT = Path("artifacts")
EXPECTED_CLASSES = {
    "image": ("ai_generated", "real"),
    "video": ("manipulated", "real"),
}


def make_transforms(training: bool) -> transforms.Compose:
    common: list[Any] = [
        transforms.Resize(256),
        transforms.RandomCrop(IMAGE_SIZE) if training else transforms.CenterCrop(IMAGE_SIZE),
    ]
    if training:
        common.append(transforms.RandomHorizontalFlip())
    common.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(common)


class ValidatedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        try:
            with Image.open(path) as image:
                sample = self.transform(image.convert("RGB")) if self.transform else image.copy()
        except (UnidentifiedImageError, OSError) as error:
            raise RuntimeError(f"Invalid training image: {path}") from error
        if self.target_transform:
            target = self.target_transform(target)
        return sample, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune an EfficientNet-B0 media detector.")
    parser.add_argument("--modality", choices=("image", "video"), required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_dataset(data_dir: Path, modality: str) -> tuple[ValidatedImageFolder, ValidatedImageFolder]:
    expected = set(EXPECTED_CLASSES[modality])
    train = ValidatedImageFolder(data_dir / "train", transform=make_transforms(training=True))
    validation = ValidatedImageFolder(data_dir / "validation", transform=make_transforms(training=False))
    if set(train.classes) != expected or set(validation.classes) != expected:
        raise ValueError(
            f"{modality} dataset classes must be {sorted(expected)} in both train and validation."
        )
    if train.classes != validation.classes:
        raise ValueError("Training and validation class order does not match.")
    if not train.samples or not validation.samples:
        raise ValueError("Both training and validation datasets must contain images.")
    return train, validation


def run_training(args: argparse.Namespace) -> dict[str, float]:
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.learning_rate <= 0:
        raise ValueError("epochs, batch size, and learning rate must be positive; workers cannot be negative.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = args.data_dir or DEFAULT_DATA_ROOT / f"{args.modality}_images"
    output_path = args.output or DEFAULT_OUTPUT_ROOT / f"{args.modality}_detector.pt"
    train_set, validation_set = validate_dataset(data_dir, args.modality)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    class_count = len(train_set.classes)
    model = build_classifier(num_classes=class_count, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        model.train()
        train_loss_total = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=pin_memory)
            targets = targets.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * targets.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, targets in validation_loader:
                images = images.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)
                predictions = model(images).argmax(dim=1)
                correct += int((predictions == targets).sum().item())
                total += int(targets.size(0))

        accuracy = correct / total
        train_loss = train_loss_total / len(train_set)
        print(
            f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.4f} "
            f"validation_accuracy={accuracy:.4f} device={device.type}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": "torchvision/efficientnet_b0",
        "modality": args.modality,
        "class_names": train_set.classes,
        "image_size": IMAGE_SIZE,
        "validation_accuracy": best_accuracy,
        "model_state_dict": best_state,
    }
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Saved best validation checkpoint to {output_path}")
    return {"validation_accuracy": best_accuracy}


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
