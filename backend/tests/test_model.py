from __future__ import annotations

import pytest
import torch
from PIL import Image

from backend.model import score_image


class ArtificialDetector:
    class config:
        id2label = {0: "artificial", 1: "real"}

    def __call__(self, **_inputs):
        return type("Output", (), {"logits": torch.tensor([[3.0, 0.0]])})()


def test_pretrained_artificial_label_maps_to_ai_generated() -> None:
    score, label = score_image(
        Image.new("RGB", (16, 16)),
        backend="transformers",
        processor=lambda **_kwargs: {},
        model=ArtificialDetector(),
        torch_module=torch,
        device=torch.device("cpu"),
    )

    assert label == "ai_generated"
    assert score > 0.9


def test_detector_with_ambiguous_labels_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one positive class"):
        score_image(
            Image.new("RGB", (16, 16)),
            backend="transformers",
            processor=lambda **_kwargs: {},
            model=type(
                "AmbiguousDetector",
                (),
                {
                    "config": type(
                        "Config",
                        (),
                        {"id2label": {0: "class_0", 1: "class_1"}},
                    )()
                },
            )(),
            torch_module=torch,
            device=torch.device("cpu"),
        )
