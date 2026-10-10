from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger("truthlense.prepare_genimage")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_ALIASES = {"train": "train", "val": "validation", "validation": "validation"}
CLASS_ALIASES = {"ai": "ai_generated", "nature": "real"}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    limit_per_class: int | None = None,
) -> dict[str, int]:
    if not source_root.is_dir():
        raise FileNotFoundError(f"GenImage root was not found: {source_root}")
    if limit_per_class is not None and limit_per_class < 1:
        raise ValueError("limit_per_class must be greater than zero.")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output_root}")

    counts = {
        f"{split}/{label}": 0
        for split in ("train", "validation")
        for label in ("ai_generated", "real")
    }
    seen_hashes: set[str] = set()

    for split_source, split_name in SPLIT_ALIASES.items():
        for class_source, class_name in CLASS_ALIASES.items():
            candidates = sorted(
                path
                for path in source_root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.parent.name.lower() == class_source
                and any(part.lower() == split_source for part in path.parts)
            )
            for source in candidates:
                count_key = f"{split_name}/{class_name}"
                if limit_per_class is not None and counts[count_key] >= limit_per_class:
                    break

                digest = file_digest(source)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)

                destination = (
                    output_root
                    / split_name
                    / class_name
                    / f"{digest}{source.suffix.lower()}"
                )
                link_or_copy(source, destination)
                counts[count_key] += 1

    if not all(counts.values()):
        raise ValueError(
            "No balanced GenImage samples were found. Expected folders named train/val "
            "and ai/nature below the supplied dataset root."
        )
    logger.info("Prepared unique GenImage samples: %s", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map GenImage train/val ai/nature folders into TruthLense training labels."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/image_images"))
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap for each class in each split; useful for CPU-only pilot runs.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_dataset(args.dataset_dir.resolve(), args.output_dir.resolve(), args.limit_per_class)


if __name__ == "__main__":
    main()
