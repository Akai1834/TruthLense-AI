from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import cv2
from PIL import Image

logger = logging.getLogger("truthlense.prepare_dfdc")
DEFAULT_OUTPUT = Path("datasets/video_images")
FRAME_COUNT = 8
FRAME_SIZE = (640, 640)


def split_for_group(group_name: str, validation_percent: int = 15) -> str:
    digest = hashlib.sha256(group_name.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    return "validation" if bucket < validation_percent else "train"


def find_metadata(dataset_root: Path) -> list[Path]:
    metadata_files = sorted(dataset_root.glob("**/metadata.json"))
    if not metadata_files:
        raise FileNotFoundError(f"No DFDC metadata.json files found under {dataset_root}")
    return metadata_files


def load_records(dataset_root: Path) -> list[tuple[Path, str, str]]:
    records: list[tuple[Path, str, str]] = []
    seen: set[Path] = set()

    for metadata_path in find_metadata(dataset_root):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read DFDC metadata file {metadata_path}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"DFDC metadata must be an object: {metadata_path}")

        for filename, details in metadata.items():
            if not isinstance(details, dict):
                raise ValueError(f"Invalid DFDC record for {filename!r} in {metadata_path}")
            relative_path = Path(filename)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe path in DFDC metadata: {filename!r}")
            video_path = (metadata_path.parent / relative_path).resolve()
            try:
                video_path.relative_to(dataset_root.resolve())
            except ValueError as error:
                raise ValueError(f"DFDC video path escapes dataset folder: {filename!r}") from error
            if not video_path.is_file() or video_path.suffix.lower() != ".mp4":
                continue
            if video_path in seen:
                continue

            raw_label = str(details.get("label", "")).upper()
            if raw_label not in {"REAL", "FAKE"}:
                raise ValueError(f"Unknown DFDC label {raw_label!r} for {filename!r}")
            label = "real" if raw_label == "REAL" else "manipulated"
            original = details.get("original")
            group = str(original) if original else video_path.name
            records.append((video_path, label, group))
            seen.add(video_path)

    if not records:
        raise FileNotFoundError(f"No DFDC videos listed in metadata were found under {dataset_root}")
    return records


def extract_video_frames(video_path: Path, output_directory: Path, frame_count: int) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        logger.warning("Skipping unreadable video %s", video_path)
        return 0

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            return 0
        indexes = sorted(
            {
                round(index * (total_frames - 1) / max(frame_count - 1, 1))
                for index in range(min(frame_count, total_frames))
            }
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        written = 0
        for frame_index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with Image.fromarray(rgb) as image:
                image.thumbnail(FRAME_SIZE)
                destination = output_directory / f"{video_path.stem}_{frame_index:08d}.jpg"
                image.save(destination, format="JPEG", quality=92, optimize=True)
                written += 1
        return written
    finally:
        capture.release()


def prepare_dataset(dataset_root: Path, output_root: Path, frame_count: int) -> tuple[int, int]:
    if frame_count < 1:
        raise ValueError("frame_count must be at least one.")
    records = load_records(dataset_root)
    train_videos = 0
    validation_videos = 0

    for index, (video_path, label, group) in enumerate(records, start=1):
        split = split_for_group(group)
        destination = output_root / split / label / video_path.stem
        written = extract_video_frames(video_path, destination, frame_count)
        if not written:
            continue
        if split == "train":
            train_videos += 1
        else:
            validation_videos += 1
        if index % 100 == 0:
            logger.info("Processed %s/%s DFDC videos", index, len(records))

    if not train_videos or not validation_videos:
        raise ValueError(
            "Frame extraction did not produce both training and validation videos. "
            "Add more data or inspect the source video files."
        )
    logger.info("Prepared %s training and %s validation videos at %s", train_videos, validation_videos, output_root)
    return train_videos, validation_videos


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DFDC video frames for frame-detector training.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames-per-video", type=int, default=FRAME_COUNT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare_dataset(args.dataset_dir.resolve(), args.output_dir.resolve(), args.frames_per_video)


if __name__ == "__main__":
    main()
