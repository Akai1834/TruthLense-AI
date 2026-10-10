from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from PIL import Image

from backend.training.prepare_dfdc import load_records, split_for_group
from backend.training.prepare_genimage import prepare_dataset
from backend.training.train_detector import validate_dataset


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color).save(path)


def test_original_and_manipulated_videos_share_group_split(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "dfdc"
    metadata_dir.mkdir()
    (metadata_dir / "original_001.mp4").touch()
    (metadata_dir / "manipulated_001.mp4").touch()
    (metadata_dir / "metadata.json").write_text(
        '{"original_001.mp4":{"label":"REAL"},'
        '"manipulated_001.mp4":{"label":"FAKE","original":"original_001.mp4"}}',
        encoding="utf-8",
    )

    records = load_records(tmp_path)
    groups = {record[2] for record in records}

    assert groups == {"original_001.mp4"}
    assert split_for_group(next(iter(groups))) in {"train", "validation"}


def test_genimage_preparation_maps_labels_and_deduplicates(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    write_image(source / "Midjourney" / "train" / "ai" / "fake.jpg", (255, 0, 0))
    duplicate = source / "ADM" / "train" / "ai" / "same_fake.jpg"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    copyfile(source / "Midjourney" / "train" / "ai" / "fake.jpg", duplicate)
    write_image(source / "Midjourney" / "train" / "nature" / "real.jpg", (0, 255, 0))
    write_image(source / "Midjourney" / "val" / "ai" / "val_fake.jpg", (0, 0, 255))
    write_image(source / "Midjourney" / "val" / "nature" / "val_real.jpg", (255, 255, 0))

    counts = prepare_dataset(source, tmp_path / "prepared")

    assert counts == {
        "train/ai_generated": 1,
        "train/real": 1,
        "validation/ai_generated": 1,
        "validation/real": 1,
    }


def test_detector_dataset_requires_matching_binary_classes(tmp_path: Path) -> None:
    for split in ("train", "validation"):
        write_image(tmp_path / split / "ai_generated" / "ai.jpg", (255, 0, 0))
        write_image(tmp_path / split / "real" / "real.jpg", (0, 255, 0))

    train, validation = validate_dataset(tmp_path, "image")

    assert train.classes == ["ai_generated", "real"]
    assert validation.classes == train.classes
