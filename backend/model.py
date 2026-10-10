from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

IMAGE_SIZE = 224
LABEL_NAMES = ("positive", "real")


def build_classifier(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def open_model_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[nn.Module, dict[int, str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint.get("class_names")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(class_names, list) or len(class_names) != 2 or not isinstance(state_dict, dict):
        raise ValueError("Model checkpoint must contain two class_names and model_state_dict.")

    model = build_classifier(num_classes=len(class_names))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, {index: str(name) for index, name in enumerate(class_names)}


def score_image(
    image: Image.Image,
    *,
    backend: str,
    processor: Any,
    model: Any,
    torch_module: Any,
    device: Any,
    labels: dict[int, str] | None = None,
) -> tuple[float, str]:
    image = image.convert("RGB")
    image.thumbnail((2048, 2048))

    if backend == "torchvision":
        inputs = evaluation_transform()(image).unsqueeze(0).to(device)
        with torch_module.inference_mode():
            logits = model(inputs)[0]
            probabilities = torch_module.softmax(logits, dim=-1).cpu().tolist()
    else:
        inputs = processor(images=image, return_tensors="pt")
        with torch_module.inference_mode():
            logits = model(**inputs).logits[0]
            probabilities = torch_module.softmax(logits, dim=-1).cpu().tolist()
        labels = {int(index): str(label) for index, label in model.config.id2label.items()}

    if labels is None:
        raise ValueError("The detector must provide its class labels.")

    normalized_labels = {
        index: label.lower().replace("-", "_").replace(" ", "_")
        for index, label in labels.items()
    }
    positive_label = next(
        (
            label
            for label in ("ai_generated", "artificial", "synthetic", "manipulated", "fake")
            if label in normalized_labels.values()
        ),
        None,
    )
    if positive_label in {"artificial", "synthetic"}:
        positive_name = "ai_generated"
    elif positive_label in {"manipulated", "fake"}:
        positive_name = "manipulated"
    else:
        positive_name = positive_label
    positive_indexes = [
        index
        for index, label in normalized_labels.items()
        if label == positive_label
    ]
    real_indexes = [
        index
        for index, label in normalized_labels.items()
        if label == "real"
    ]
    if len(labels) != 2 or len(positive_indexes) != 1 or len(real_indexes) != 1:
        raise ValueError("Model must have exactly one positive class and one real class.")
    return float(probabilities[positive_indexes[0]]), str(positive_name)
