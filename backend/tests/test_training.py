from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from shutil import copyfile

from PIL import Image
import pytest

from backend.training.prepare_dfdc import load_records, split_for_group
from backend.training.prepare_celeba import prepare_dataset as prepare_celeba_dataset
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


def test_celeba_preparation_uses_train_and_validation_but_excludes_test(
    tmp_path: Path,
) -> None:
    source = tmp_path / "celeba"
    source.mkdir()
    partition_path = source / "list_eval_partition.csv"
    with partition_path.open("w", newline="", encoding="utf-8") as partition_file:
        writer = csv.writer(partition_file)
        writer.writerow(("image_id", "partition"))
        writer.writerows(
            (
                ("000001.jpg", "0"),
                ("000002.jpg", "1"),
                ("000003.jpg", "2"),
            )
        )

    archive_path = source / "archive.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for image_id in ("000001.jpg", "000002.jpg", "000003.jpg"):
            archive.writestr(
                f"img_align_celeba/img_align_celeba/{image_id}",
                f"image:{image_id}",
            )

    output = tmp_path / "prepared"
    counts = prepare_celeba_dataset(source, output)

    assert counts == {"train/real": 1, "validation/real": 1}
    assert (output / "train" / "real" / "000001.jpg").read_bytes() == b"image:000001.jpg"
    assert (output / "validation" / "real" / "000002.jpg").exists()
    assert not (output / "test").exists()
    with pytest.raises(ValueError, match="dataset classes must be"):
        validate_dataset(output, "image")


def test_detector_dataset_requires_matching_binary_classes(tmp_path: Path) -> None:
    for split in ("train", "validation"):
        write_image(tmp_path / split / "ai_generated" / "ai.jpg", (255, 0, 0))
        write_image(tmp_path / split / "real" / "real.jpg", (0, 255, 0))

    train, validation = validate_dataset(tmp_path, "image")

    assert train.classes == ["ai_generated", "real"]
    assert validation.classes == train.classes
