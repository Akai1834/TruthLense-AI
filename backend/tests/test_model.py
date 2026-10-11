from __future__ import annotations

import json

import pytest
import torch
from PIL import Image

from backend.model import open_huggingface_deepfake_model, score_image


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


class TimmDeepfakeDetector:
    def __call__(self, _inputs):
        return torch.tensor([[2.0]])


def test_timm_deepfake_model_maps_sigmoid_score_to_ai_generated() -> None:
    score, label = score_image(
        Image.new("RGB", (16, 16)),
        backend="timm_deepfake",
        processor=None,
        model=TimmDeepfakeDetector(),
        torch_module=torch,
        device=torch.device("cpu"),
        labels={0: "ai_generated", 1: "real"},
        image_size=380,
    )

    assert label == "ai_generated"
    assert score == pytest.approx(0.8808, abs=0.0001)


def test_huggingface_checkpoint_loader_builds_configured_timm_model(monkeypatch, tmp_path) -> None:
    import timm
    import huggingface_hub

    model_id = "test/deepfake-detector"
    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "pytorch_model.bin"
    config_path.write_text(
        json.dumps({"model_name": "mobilenetv3_small_050", "image_size": 64}),
        encoding="utf-8",
    )
    source_model = timm.create_model(
        "mobilenetv3_small_050",
        pretrained=False,
        num_classes=1,
    )
    torch.save(source_model.state_dict(), weights_path)
    downloaded = {"config.json": str(config_path), "pytorch_model.bin": str(weights_path)}

    def download(repo_id, filename, *, token):
        assert repo_id == model_id
        assert token is None
        return downloaded[filename]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    model, image_size = open_huggingface_deepfake_model(
        model_id,
        torch.device("cpu"),
    )

    assert image_size == 64
    assert model.training is False
    assert set(model.state_dict()) == set(source_model.state_dict())
