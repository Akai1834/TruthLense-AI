from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from .model import open_model_checkpoint, score_image

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("truthlense")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MAX_VIDEO_SECONDS = 300
VIDEO_FRAME_SAMPLES = 8
DEFAULT_MODEL_ID = "haywoodsloan/ai-image-detector-deploy"
HF_IMAGE_MODEL_ID = os.getenv(
    "HF_IMAGE_MODEL_ID",
    "Arko007/deepfake-image-detector",
)
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models"
HF_TOKEN = os.getenv("HF_TOKEN", "")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_MODEL_PATH = Path(
    os.getenv("IMAGE_MODEL_PATH", str(PROJECT_ROOT / "artifacts" / "image_detector.pt"))
).resolve()
VIDEO_MODEL_PATH = Path(
    os.getenv("VIDEO_MODEL_PATH", str(PROJECT_ROOT / "artifacts" / "video_detector.pt"))
).resolve()
MODEL_ID = os.getenv("MODEL_ID", DEFAULT_MODEL_ID)
VIDEO_MODEL_ID = os.getenv("VIDEO_MODEL_ID", MODEL_ID)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
ANALYSIS_TABLE = "analysis_results"

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    access_token: str


_models: dict[str, dict[str, Any]] = {}
_model_locks = {"image": asyncio.Lock(), "video": asyncio.Lock()}
_inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    _models.clear()


app = FastAPI(
    title="TruthLense private media analysis API",
    version="0.2.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5500,http://127.0.0.1:5500",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "apikey", "Prefer"],
)


async def require_user(authorization: str | None = Header(default=None)) -> AuthenticatedUser:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in before requesting this resource.")

    access_token = authorization[7:].strip()
    if not access_token:
        raise HTTPException(status_code=401, detail="Sign in before requesting this resource.")
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=503,
            detail="Authentication service is not configured on the analysis API.",
        )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {access_token}",
                },
            )
    except httpx.RequestError as error:
        logger.warning("Could not verify an API request with Supabase: %s", error)
        raise HTTPException(
            status_code=503,
            detail="Could not verify your sign-in with the authentication service.",
        ) from error

    if response.status_code in (401, 403):
        raise HTTPException(status_code=401, detail="Your session expired. Please sign in again.")
    if response.status_code != 200:
        logger.error("Supabase user verification returned HTTP %s", response.status_code)
        raise HTTPException(
            status_code=503,
            detail="The authentication service could not verify this request.",
        )

    try:
        user = response.json()
    except ValueError as error:
        logger.error("Supabase returned invalid user data.")
        raise HTTPException(
            status_code=503,
            detail="Authentication returned an invalid response.",
        ) from error

    if not isinstance(user, dict) or not isinstance(user.get("id"), str):
        raise HTTPException(status_code=401, detail="The sign-in session is not valid.")
    return AuthenticatedUser(user_id=user["id"], access_token=access_token)


async def get_detector(media_type: str) -> dict[str, Any]:
    if media_type not in {"image", "video"}:
        raise ValueError("Unsupported detector media type.")
    cached = _models.get(media_type)
    if cached is not None:
        return cached

    async with _model_locks[media_type]:
        cached = _models.get(media_type)
        if cached is not None:
            return cached

        checkpoint_path = IMAGE_MODEL_PATH if media_type == "image" else VIDEO_MODEL_PATH
        if media_type == "image" and HF_TOKEN and not checkpoint_path.is_file():
            bundle = {
                "backend": "huggingface_api",
                "model_id": HF_IMAGE_MODEL_ID,
                "endpoint": f"{HF_INFERENCE_URL}/{HF_IMAGE_MODEL_ID}",
            }
            _models[media_type] = bundle
            return bundle

        try:
            import torch
        except ImportError as error:
            logger.exception("Install the backend requirements to enable model inference.")
            raise HTTPException(
                status_code=503,
                detail="The analysis model dependencies are not installed on the server.",
            ) from error
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if media_type == "video" and not checkpoint_path.is_file():
            raise HTTPException(
                status_code=503,
                detail="The video detector has not been trained yet. Prepare the DFDC data and train its checkpoint.",
            )

        if checkpoint_path.is_file():
            try:
                from .model import open_model_checkpoint

                model, labels = await run_in_threadpool(
                    open_model_checkpoint,
                    checkpoint_path,
                    device,
                )
            except (OSError, RuntimeError, ValueError) as error:
                logger.exception("Could not load the %s model checkpoint.", media_type)
                raise HTTPException(
                    status_code=503,
                    detail=f"The {media_type} detector checkpoint could not be loaded.",
                ) from error
            expected_positive = "ai_generated" if media_type == "image" else "manipulated"
            if sorted(labels.values()) != sorted((expected_positive, "real")):
                raise HTTPException(
                    status_code=503,
                    detail=f"The {media_type} model checkpoint has incorrect class labels.",
                )
            bundle = {
                "backend": "torchvision",
                "processor": None,
                "model": model,
                "torch": torch,
                "device": device,
                "labels": labels,
                "model_id": f"torchvision/efficientnet_b0:{checkpoint_path.name}",
            }
        else:
            try:
                from transformers import AutoImageProcessor, AutoModelForImageClassification

                processor = await run_in_threadpool(
                    AutoImageProcessor.from_pretrained,
                    MODEL_ID if media_type == "image" else VIDEO_MODEL_ID,
                )
                model_id = MODEL_ID if media_type == "image" else VIDEO_MODEL_ID
                model = await run_in_threadpool(
                    AutoModelForImageClassification.from_pretrained,
                    model_id,
                )
                model.to(device)
                model.eval()
            except ImportError as error:
                logger.exception("Install transformers to use the default image detector.")
                raise HTTPException(
                    status_code=503,
                    detail="The default image detector dependencies are not installed.",
                ) from error
            except (OSError, ValueError, RuntimeError) as error:
                logger.exception("Could not load model %s.", model_id)
                raise HTTPException(
                    status_code=503,
                    detail="The detector model could not be loaded. Check model access and server memory.",
                ) from error
            bundle = {
                "backend": "transformers",
                "processor": processor,
                "model": model,
                "torch": torch,
                "device": device,
                "labels": {
                    int(index): str(label)
                    for index, label in model.config.id2label.items()
                },
                "model_id": model_id,
            }

        _models[media_type] = bundle
        return bundle


def classify_image(image: Image.Image, detector: dict[str, Any], media_type: str) -> float:
    score, score_label = score_image(
        image,
        backend=detector["backend"],
        processor=detector["processor"],
        model=detector["model"],
        torch_module=detector["torch"],
        device=detector["device"],
        labels=detector["labels"],
    )
    expected_label = "ai_generated" if media_type == "image" else "manipulated"
    if score_label != expected_label:
        raise HTTPException(
            status_code=503,
            detail=f"The loaded {media_type} detector does not return the required class labels.",
        )
    return score


def analyze_image_bytes(data: bytes, detector: dict[str, Any], media_type: str) -> float:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return classify_image(image, detector, media_type)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status_code=415,
            detail="The selected file is not a supported image.",
        ) from error


def parse_huggingface_image_scores(payload: Any) -> float:
    if not isinstance(payload, list) or len(payload) != 2:
        raise HTTPException(
            status_code=503,
            detail="The Hugging Face detector returned an unsupported result.",
        )

    fake_labels = {"fake", "deepfake", "manipulated", "ai_generated", "synthetic"}
    real_labels = {"real", "authentic", "genuine"}
    scores: dict[str, float] = {}
    for item in payload:
        if not isinstance(item, dict):
            break
        label = item.get("label")
        score = item.get("score")
        if not isinstance(label, str) or isinstance(score, bool) or not isinstance(score, (int, float)):
            break
        normalized_label = label.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized_label in fake_labels:
            category = "fake"
        elif normalized_label in real_labels:
            category = "real"
        else:
            break
        if category in scores or not math.isfinite(score) or not 0 <= score <= 1:
            break
        scores[category] = float(score)
    else:
        if set(scores) == {"fake", "real"}:
            return scores["fake"]

    raise HTTPException(
        status_code=503,
        detail="The Hugging Face detector did not return one fake and one real score.",
    )


def validate_image_bytes(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(
            status_code=415,
            detail="The selected file is not a supported image.",
        ) from error


async def analyze_image_with_huggingface(data: bytes, detector: dict[str, Any]) -> float:
    validate_image_bytes(data)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=10)
        ) as client:
            response = await client.post(
                detector["endpoint"],
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/octet-stream",
                },
                content=data,
            )
    except httpx.RequestError as error:
        logger.warning(
            "Could not reach Hugging Face model %s: %s",
            detector["model_id"],
            error,
        )
        raise HTTPException(
            status_code=503,
            detail="Could not reach the Hugging Face image detector.",
        ) from error

    if response.status_code in (401, 403):
        logger.error("Hugging Face rejected access to model %s.", detector["model_id"])
        raise HTTPException(
            status_code=503,
            detail="Hugging Face rejected the configured token or model access.",
        )
    if response.status_code == 429:
        raise HTTPException(
            status_code=503,
            detail="The Hugging Face image detector is rate-limited. Please try again later.",
        )
    if response.status_code == 503:
        raise HTTPException(
            status_code=503,
            detail="The Hugging Face image detector is loading or temporarily unavailable. Please retry shortly.",
        )
    if response.status_code != 200:
        logger.error(
            "Hugging Face model %s returned HTTP %s.",
            detector["model_id"],
            response.status_code,
        )
        raise HTTPException(
            status_code=503,
            detail="The Hugging Face image detector request failed.",
        )

    try:
        payload = response.json()
    except ValueError as error:
        logger.error("Hugging Face model %s returned invalid JSON.", detector["model_id"])
        raise HTTPException(
            status_code=503,
            detail="The Hugging Face image detector returned an invalid response.",
        ) from error
    return parse_huggingface_image_scores(payload)


def analyze_video_bytes(
    data: bytes,
    suffix: str,
    detector: dict[str, Any],
) -> tuple[float, int]:
    try:
        import cv2
    except ImportError as error:
        logger.exception("Install opencv-python-headless to enable video analysis.")
        raise HTTPException(
            status_code=503,
            detail="Video analysis dependencies are not installed.",
        ) from error

    temp_path: str | None = None
    capture = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(data)
            temp_path = temporary_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise HTTPException(status_code=415, detail="The selected video could not be decoded.")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if frame_count <= 0:
            raise HTTPException(status_code=415, detail="The video contains no readable frames.")
        duration = frame_count / fps if fps > 0 else 0
        if duration > MAX_VIDEO_SECONDS:
            raise HTTPException(
                status_code=413,
                detail=f"Videos must be {MAX_VIDEO_SECONDS // 60} minutes or shorter.",
            )

        sample_count = min(VIDEO_FRAME_SAMPLES, frame_count)
        frame_indexes = [
            round(index * (frame_count - 1) / max(sample_count - 1, 1))
            for index in range(sample_count)
        ]
        scores: list[float] = []
        for frame_index in frame_indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with Image.fromarray(rgb_frame) as image:
                scores.append(classify_image(image, detector, "video"))

        if not scores:
            raise HTTPException(status_code=415, detail="No readable video frames were found.")
        return mean(scores), len(scores)
    except cv2.error as error:
        logger.exception("Video decoding failed.")
        raise HTTPException(status_code=415, detail="The selected video could not be decoded.") from error
    finally:
        if capture is not None:
            capture.release()
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                logger.exception("Could not remove the temporary video file.")


async def read_upload(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total_size = 0
    while chunk := await upload.read(1024 * 1024):
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Files must be 25 MB or smaller.")
        chunks.append(chunk)
    if not total_size:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return b"".join(chunks)


async def save_analysis(user: AuthenticatedUser, record: dict[str, Any]) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{SUPABASE_URL}/rest/v1/{ANALYSIS_TABLE}",
                params={"select": "*"},
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {user.access_token}",
                    "Content-Type": "application/json",
                    "Prefer": "return=representation",
                },
                json=record,
            )
    except httpx.RequestError as error:
        logger.warning("Could not save analysis results to Supabase: %s", error)
        raise HTTPException(
            status_code=503,
            detail="Analysis finished but its private result could not be saved. Please retry.",
        ) from error

    if response.status_code not in (200, 201):
        logger.error("Supabase analysis insert returned HTTP %s", response.status_code)
        if response.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Your session expired. Please sign in again.")
        if response.status_code in (404, 406):
            raise HTTPException(
                status_code=503,
                detail="The analysis_results table is not installed yet. Apply the Supabase migration.",
            )
        raise HTTPException(
            status_code=503,
            detail="Analysis finished but its private result could not be saved.",
        )

    try:
        rows = response.json()
    except ValueError as error:
        logger.error("Supabase returned invalid data after saving an analysis.")
        raise HTTPException(
            status_code=503,
            detail="Analysis finished but the saved result could not be read.",
        ) from error
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise HTTPException(
            status_code=503,
            detail="Analysis finished but Supabase did not return the saved result.",
        )
    return rows[0]


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "image_model": {
            "state": (
                "trained"
                if IMAGE_MODEL_PATH.is_file()
                else "Hugging Face Inference API"
                if HF_TOKEN
                else "pretrained fallback"
            ),
            "path": (
                str(IMAGE_MODEL_PATH)
                if IMAGE_MODEL_PATH.is_file()
                else HF_IMAGE_MODEL_ID
                if HF_TOKEN
                else MODEL_ID
            ),
        },
        "video_model": {
            "state": "trained" if VIDEO_MODEL_PATH.is_file() else "not trained",
            "path": str(VIDEO_MODEL_PATH) if VIDEO_MODEL_PATH.is_file() else None,
        },
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_ANON_KEY),
    }


@app.get("/api/analyses")
async def list_analyses(user: AuthenticatedUser = Depends(require_user)) -> list[dict[str, Any]]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/{ANALYSIS_TABLE}",
                params={
                    "select": "id,media_type,positive_percent,real_percent,frames_analyzed,model_id,created_at",
                    "order": "created_at.desc",
                    "limit": "20",
                },
                headers={
                    "apikey": SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {user.access_token}",
                },
            )
    except httpx.RequestError as error:
        logger.warning("Could not load private analysis history: %s", error)
        raise HTTPException(status_code=503, detail="Could not load your analysis history.") from error

    if response.status_code != 200:
        logger.error("Supabase analysis history returned HTTP %s", response.status_code)
        if response.status_code in (401, 403):
            raise HTTPException(status_code=401, detail="Your session expired. Please sign in again.")
        if response.status_code in (404, 406):
            raise HTTPException(
                status_code=503,
                detail="The analysis_results table is not installed yet. Apply the Supabase migration.",
            )
        raise HTTPException(status_code=503, detail="Could not load your analysis history.")
    try:
        rows = response.json()
    except ValueError as error:
        logger.error("Supabase returned invalid analysis history.")
        raise HTTPException(status_code=503, detail="Could not read your analysis history.") from error
    if not isinstance(rows, list):
        raise HTTPException(status_code=503, detail="Supabase returned an invalid analysis history.")
    return rows


@app.post("/api/analyze")
async def analyze_media(
    file: UploadFile = File(...),
    user: AuthenticatedUser = Depends(require_user),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    video_extensions = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}
    declared_video = (file.content_type or "").startswith("video/")
    declared_image = (file.content_type or "").startswith("image/")
    if not declared_image and not declared_video:
        raise HTTPException(status_code=415, detail="Upload a supported image or video file.")
    if declared_video and suffix not in video_extensions:
        raise HTTPException(status_code=415, detail="This video format is not supported.")

    try:
        data = await read_upload(file)
    finally:
        await file.close()

    media_type = "video" if declared_video else "image"
    detector = await get_detector(media_type)
    async with _inference_lock:
        if declared_video:
            positive_score, analyzed_frames = await run_in_threadpool(
                analyze_video_bytes,
                data,
                suffix,
                detector,
            )
            score_label = "manipulated"
        else:
            if detector["backend"] == "huggingface_api":
                positive_score = await analyze_image_with_huggingface(data, detector)
            else:
                positive_score = await run_in_threadpool(
                    analyze_image_bytes,
                    data,
                    detector,
                    media_type,
                )
            analyzed_frames = 1
            score_label = "ai_generated"

    positive_percent = round(max(0.0, min(1.0, positive_score)) * 100, 1)
    result = {
        "media_type": media_type,
        "score_label": score_label,
        "positive_percent": positive_percent,
        "real_percent": round(100 - positive_percent, 1),
        "frames_analyzed": analyzed_frames,
        "model_id": detector["model_id"],
        "disclaimer": (
            "These are uncalibrated model scores, not verified probabilities or proof. "
            "This is not a source-authenticity detector. Compression, edits, unseen generators, "
            "and real-world content can produce errors. Video scores average sampled frames and "
            "may miss manipulated segments."
        ),
    }
    stored_result = await save_analysis(
        user,
        {
            "user_id": user.user_id,
            "media_type": result["media_type"],
            "score_label": result["score_label"],
            "positive_percent": result["positive_percent"],
            "real_percent": result["real_percent"],
            "frames_analyzed": result["frames_analyzed"],
            "model_id": result["model_id"],
        },
    )
    return {
        **result,
        "id": stored_result["id"],
        "created_at": stored_result["created_at"],
    }
