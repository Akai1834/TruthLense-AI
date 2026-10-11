from __future__ import annotations

import argparse
import csv
import logging
import re
import tempfile
import time
import zipfile
from pathlib import Path
from shutil import copyfileobj

logger = logging.getLogger("truthlense.prepare_celeba")
IMAGE_MEMBER_PREFIX = "img_align_celeba/img_align_celeba/"
IMAGE_NAME = re.compile(r"\d{6}\.jpg")
PARTITION_SPLITS = {"0": "train", "1": "validation"}


def load_partitions(partition_path: Path) -> dict[str, str]:
    partitions: dict[str, str] = {}
    with partition_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not {"image_id", "partition"}.issubset(reader.fieldnames):
            raise ValueError(f"CelebA partition CSV must contain image_id and partition: {partition_path}")
        for row in reader:
            image_id = row["image_id"]
            partition = row["partition"]
            if not IMAGE_NAME.fullmatch(image_id) or partition not in {"0", "1", "2"}:
                raise ValueError(f"Invalid CelebA partition row: {row}")
            if image_id in partitions:
                raise ValueError(f"Duplicate CelebA image in partition CSV: {image_id}")
            if partition in PARTITION_SPLITS:
                partitions[image_id] = PARTITION_SPLITS[partition]
    if not partitions:
        raise ValueError(f"No CelebA train or validation rows found in {partition_path}")
    return partitions


def prepare_dataset(dataset_root: Path, output_root: Path) -> dict[str, int]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"CelebA dataset directory was not found: {dataset_root}")

    partition_path = dataset_root / "list_eval_partition.csv"
    if not partition_path.is_file():
        raise FileNotFoundError(f"CelebA partition CSV was not found: {partition_path}")
    partitions = load_partitions(partition_path)

    extracted_images = dataset_root / "img_align_celeba" / "img_align_celeba"
    archive_path = dataset_root / "archive.zip"
    image_archive: zipfile.ZipFile | None = None
    archive_members: dict[str, zipfile.ZipInfo] = {}
    if extracted_images.is_dir():
        missing = [
            image_id
            for image_id in partitions
            if not (extracted_images / image_id).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"CelebA image is missing: {missing[0]}")
    elif archive_path.is_file():
        image_archive = zipfile.ZipFile(archive_path)
        for member in image_archive.infolist():
            if not member.is_dir() and member.filename.startswith(IMAGE_MEMBER_PREFIX):
                image_id = member.filename.removeprefix(IMAGE_MEMBER_PREFIX)
                if image_id in partitions:
                    archive_members[image_id] = member
        missing = [image_id for image_id in partitions if image_id not in archive_members]
        if missing:
            image_archive.close()
            raise FileNotFoundError(f"CelebA archive image is missing: {missing[0]}")
    else:
        raise FileNotFoundError(
            f"Expected extracted images at {extracted_images} or an archive at {archive_path}"
        )

    if output_root.exists():
        if any(output_root.iterdir()):
            if image_archive is not None:
                image_archive.close()
            raise FileExistsError(f"Output folder is not empty: {output_root}")
        output_root.rmdir()

    counts = {"train/real": 0, "validation/real": 0}
    output_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            dir=output_root.parent,
            prefix=f".{output_root.name}.",
        ) as temporary_directory:
            staged_root = Path(temporary_directory)
            for image_id, split in partitions.items():
                destination = staged_root / split / "real" / image_id
                destination.parent.mkdir(parents=True, exist_ok=True)
                if image_archive is None:
                    with (extracted_images / image_id).open("rb") as source:
                        with destination.open("wb") as target:
                            copyfileobj(source, target)
                else:
                    with image_archive.open(archive_members[image_id]) as source:
                        with destination.open("wb") as target:
                            copyfileobj(source, target)
                counts[f"{split}/real"] += 1
                if sum(counts.values()) % 25_000 == 0:
                    logger.info("Prepared %s/%s CelebA images", sum(counts.values()), len(partitions))
            for attempt in range(3):
                try:
                    staged_root.rename(output_root)
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    time.sleep(1)
    finally:
        if image_archive is not None:
            image_archive.close()

    logger.info("Prepared CelebA real images: %s", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CelebA train and validation images as the real class."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/image_images"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_dataset(args.dataset_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
