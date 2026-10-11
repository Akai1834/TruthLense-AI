from __future__ import annotations

from fastapi import HTTPException
import pytest
import torch
from PIL import Image

from backend.app import parse_huggingface_image_scores
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


def test_huggingface_fake_score_is_returned_regardless_of_label_order() -> None:
    assert parse_huggingface_image_scores(
        [
            {"label": "REAL", "score": 0.2},
            {"label": "FAKE", "score": 0.8},
        ]
    ) == 0.8


def test_huggingface_scores_reject_unknown_labels() -> None:
    with pytest.raises(HTTPException) as exc_info:
        parse_huggingface_image_scores(
            [
                {"label": "LABEL_0", "score": 0.2},
                {"label": "LABEL_1", "score": 0.8},
            ]
        )
    assert "one fake and one real score" in exc_info.value.detail
