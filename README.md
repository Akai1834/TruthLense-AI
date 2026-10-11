# TruthLense AI

TruthLense is a static frontend, a FastAPI inference service, and separate image/video-frame training pipelines. Analysis scores are model estimates, not proof of authenticity.

## Project layout and virtual environment

- `.venv/` contains Python and installed third-party packages only. Do not put application code, datasets, credentials, or model checkpoints inside it.
- `backend/` contains the FastAPI app, shared model code, requirements, and data-preparation/training scripts.
- `supabase/migrations/` contains SQL schema changes to apply to Supabase.
- `datasets/` contains your downloaded/prepared training data. It is ignored by Git.
- `artifacts/` contains trained checkpoints. It is ignored by Git.
- `config.js` contains the frontend Supabase project URL and public publishable key. Never put a service-role/secret key here.

## Install Python packages into the project environment

The existing project `.venv` uses Python 3.14.7 and has no backend/ML dependencies installed. This PC has an Intel UHD integrated GPU, so use the CPU-only PyTorch wheels; training will be slow and is best limited to a small pilot.

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

The first command installs PyTorch and torchvision into `.venv`; the second installs FastAPI, video/image processing, Transformers fallback, and training/test dependencies. The CPU PyTorch wheel was verified to exist for this Python version. Do not copy downloaded package source files into the project.

## Connect Supabase

1. The frontend uses the project URL and publishable key in `config.js`; those are public browser settings. The FastAPI backend uses the same values from environment variables.
2. In Supabase Dashboard → **SQL Editor**, run [`supabase/migrations/202610080001_analysis_results.sql`](./supabase/migrations/202610080001_analysis_results.sql). It creates an `analysis_results` table with row-level security. Signed-in users can select/insert only their own rows.
3. The application stores only analysis scores/model metadata in Supabase—not the uploaded media. The backend verifies each access token with Supabase Auth and sends inserts using that same user's token, so RLS applies. There is no public upload bucket/feed.
4. Set the allowed redirect URLs in Supabase Auth to include `http://localhost:5500/index.html` and `http://127.0.0.1:5500/index.html`.

## Run locally on Windows

Open two PowerShell windows from the project root.

Frontend:

```powershell
.\.venv\Scripts\python.exe -m http.server 5500
```

Backend:

```powershell
$env:SUPABASE_URL = "https://YOUR-PROJECT.supabase.co"
$env:SUPABASE_ANON_KEY = "YOUR_SUPABASE_PUBLISHABLE_OR_ANON_KEY"
$env:ALLOWED_ORIGINS = "http://localhost:5500,http://127.0.0.1:5500"
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:5500`. The backend downloads the public `Arko007/deepfake-image-detector` PyTorch checkpoint on the first image analysis (about 90 MB) and caches it for later requests. No Hugging Face token is required for this public repository; if you use a gated model, set `HF_TOKEN` only in the backend environment. Never put tokens in `frontend/config.js`, HTML, or browser JavaScript. You can override the model repo with `HF_IMAGE_MODEL_ID`. A trained local image checkpoint takes precedence. Keep other server-only configuration in environment variables; never use a Supabase service-role key in the frontend or inference service. Video analysis returns a clear not-trained response until a video checkpoint exists.

## Training data

Training requires labeled data; **no model has been trained yet**. Do not put dataset archives or checkpoints inside `.venv`.

### AI-generated image detector

Download the official [GenImage dataset](https://github.com/GenImage-Dataset/GenImage) or a smaller labeled subset. Its documented layout has generator folders with `train/ai`, `train/nature`, `val/ai`, and `val/nature`. The dataset uses a **non-commercial CC BY-NC-SA 4.0 license with additional dataset terms**; check those terms before using it beyond non-commercial academic/research work. Keep the source archives/data outside Git.

Prepare a pilot of up to 1,000 unique images per class per split:

```powershell
.\.venv\Scripts\python.exe -m backend.training.prepare_genimage --dataset-dir "D:\datasets\GenImage" --output-dir datasets\image_images --limit-per-class 1000
.\.venv\Scripts\python.exe -m backend.training.train_detector --modality image --data-dir datasets\image_images --output artifacts\image_detector.pt --epochs 3 --batch-size 4
```

For a real benchmark, use the full permitted training data and keep the official validation split untouched. Results on a small CIFAKE/CIFAR-size subset would only validate the pipeline, not establish real-world detection performance.

### Manipulated-video detector

Request the [official DFDC Preview dataset](https://ai.meta.com/datasets/dfdc/) first (5,000 videos); full DFDC is 124,000 videos. Meta requires an AWS account, an IAM user/access keys, and the AWS account number to access it. **Do not send AWS credentials to me or store them in this repo.** Keep the downloaded video archives outside `.venv` and Git.

Extract the preview archives into a directory that preserves each chunk's `metadata.json` next to its MP4s, then run:

```powershell
.\.venv\Scripts\python.exe -m backend.training.prepare_dfdc --dataset-dir "D:\datasets\dfdc_preview" --output-dir datasets\video_images --frames-per-video 8
.\.venv\Scripts\python.exe -m backend.training.train_detector --modality video --data-dir datasets\video_images --output artifacts\video_detector.pt --epochs 3 --batch-size 2
```

The preparation step splits by original-video group before extracting frames so related real/fake videos are not divided between train and validation. This first video detector is a frame classifier whose scores are averaged across up to eight sampled frames; it is **not** a temporal video model and can miss short manipulations.

Training uses ImageNet-pretrained EfficientNet-B0 weights, downloaded by torchvision the first time, then saves best-validation checkpoints under `artifacts/`. Inference automatically loads `artifacts/image_detector.pt` and `artifacts/video_detector.pt` when available. Checkpoints and datasets are intentionally not committed.

## Results and limits

`POST /api/analyze` requires a valid signed-in Supabase token and returns model estimates. Each result is persisted to the authenticated user's own row in Supabase and displayed in that user's private history. The API does not store original media; video decoding uses a temporary file removed after processing.

The image detector is [`Arko007/deepfake-image-detector`](https://huggingface.co/Arko007/deepfake-image-detector), an EfficientNetV2-S PyTorch checkpoint loaded locally with `timm`. Its model card describes it as experimental, notes missing training-data provenance, and warns performance may degrade on full-frame images or unseen domains. The checkpoint is downloaded from Hugging Face on first use, so the backend needs network access and enough disk space for the model cache. The trained image and video-frame models are separate. Scores are model estimates, not proof of authenticity; compression, unseen generators, editing, and distribution shift can cause errors. Never use this output alone for high-impact decisions.

Uploads are limited to 25 MB; videos are limited to five minutes and eight sampled frames. For public deployment, use HTTPS, an HTTPS API URL in `config.js`, exact production CORS origins, rate limiting, and platform body-size/time limits. For production ML training, use a GPU host—this development PC has no CUDA-capable GPU.
#