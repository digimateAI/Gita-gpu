# 🕉️ Gita Guru — From Local Script to Live RunPod API

A complete guide on how we converted a local Bhagavad Gita chatbot (`gita_chatbot.py`) into a live, cloud-hosted serverless API using **RunPod Serverless** and **Docker**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Step 1: Understanding the Local Script](#step-1-understanding-the-local-script)
- [Step 2: Creating the RunPod Handler](#step-2-creating-the-runpod-handler)
- [Step 3: Creating the Dockerfile](#step-3-creating-the-dockerfile)
- [Step 4: Supporting Files](#step-4-supporting-files)
- [Step 5: Build & Push Docker Image](#step-5-build--push-docker-image)
- [Step 6: Deploy on RunPod](#step-6-deploy-on-runpod)
- [Step 7: Testing the API](#step-7-testing-the-api)
- [Bugs Encountered & Fixes](#bugs-encountered--fixes)
- [API Reference](#api-reference)
- [Cost Estimate](#cost-estimate)

---

## Overview

| Component | Detail |
|---|---|
| **Model** | Llama 3.1 8B Instruct (4-bit quantized) |
| **Fine-tuning** | LoRA adapters trained on Bhagavad Gita content |
| **Framework** | Unsloth + HuggingFace Transformers + PEFT |
| **Deployment** | RunPod Serverless (GPU: 24 GB VRAM) |
| **Docker Image** | `digimate2023/gita-guru-serverless:latest` |
| **Endpoint** | `https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync` |

---

## Architecture

```
┌─────────────────┐     POST /runsync      ┌──────────────────────────┐
│   Frontend /    │ ──────────────────────► │   RunPod Serverless      │
│   Postman /     │                         │                          │
│   Any Client    │ ◄────────────────────── │  ┌────────────────────┐  │
│                 │     JSON Response       │  │   Docker Container │  │
└─────────────────┘                         │  │                    │  │
                                            │  │  handler.py        │  │
                                            │  │  LoRA Adapters     │  │
                                            │  │  + Base Model      │  │
                                            │  │    (auto-download) │  │
                                            │  └────────────────────┘  │
                                            └──────────────────────────┘
```

**How it works:**
1. Client sends a POST request with a question
2. RunPod spins up a GPU worker (if none running)
3. On first boot (cold-start), the base Llama 3.1 model is downloaded from HuggingFace (~4.5 GB)
4. LoRA adapters (baked into the Docker image) are loaded on top of the base model
5. The model generates a response using the Gita Guru persona
6. Response is cleaned and returned as JSON

---

## Project Structure

```
D:\Gita_FineTuning_Atanu\
├── gita_chatbot.py                 # Original local chatbot (terminal-based)
├── handler.py                      # RunPod serverless handler (API version)
├── Dockerfile                      # Docker image configuration
├── requirements.txt                # Python dependencies
├── .dockerignore                   # Files excluded from Docker build
├── test_input.json                 # Local testing payload for RunPod SDK
├── README.md                       # This file
└── gita_model_output/
    └── lora_adapters/
        ├── adapter_config.json     # LoRA configuration
        ├── adapter_model.safetensors  # Fine-tuned weights (~320 MB)
        ├── tokenizer.json          # Tokenizer vocabulary
        ├── tokenizer_config.json   # Tokenizer settings
        ├── special_tokens_map.json # Special token definitions
        └── chat_template.jinja     # Chat template
```

---

## Step 1: Understanding the Local Script

The original `gita_chatbot.py` runs interactively in the terminal. It:
- Loads the fine-tuned model using Unsloth
- Takes user input via `input()` 
- Formats prompts using ChatML format
- Streams responses to the terminal
- Maintains conversation history in memory

**Key components that needed to be ported:**
- System prompt (the Gita Guru persona)
- Generation config (temperature, top_p, etc.)
- ChatML prompt formatting
- Stopping criteria (to cut off at turn boundaries)
- Response cleanup (remove ChatML artifacts)

---

## Step 2: Creating the RunPod Handler

### What Changed from Local → API

| Feature | `gita_chatbot.py` (Local) | `handler.py` (RunPod) |
|---|---|---|
| **Input** | `input()` from terminal | JSON payload via HTTP |
| **Output** | Streamed `print()` | JSON response `{"response": "..."}` |
| **History** | In-memory list | Passed in request (stateless) |
| **Model loading** | On script start | Once at cold-start (module level) |
| **Error handling** | Print and continue | Return `{"error": "..."}` |
| **Streaming** | `RealTimeGuruStreamer` | Removed (not applicable for API) |

### Key Design Decisions in `handler.py`

**1. Model loads at module level (not inside the handler function)**
```python
# This runs ONCE when the worker boots
model, tokenizer = FastLanguageModel.from_pretrained(...)
FastLanguageModel.for_inference(model)

def handler(job):
    # This runs per-request, model is already loaded
    ...
```

**2. Stateless design — history is passed in the request**
```python
history = job_input.get("history", [])  # Client manages conversation state
```

**3. Per-request generation parameter overrides**
```python
for key in ("max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty"):
    if key in job_input:
        gen_config[key] = job_input[key]
```

**4. Logging fix — monkey-patching Python's `LogRecord.getMessage`**

The `transformers` library v5.2.0 has a bug where `logger.warning_once(message, FutureWarning)` passes `FutureWarning` as a formatting argument, causing `TypeError: not all arguments converted during string formatting`. We fix this by patching the exact method that crashes:

```python
_original_getMessage = logging.LogRecord.getMessage

def _safe_getMessage(self):
    try:
        return _original_getMessage(self)
    except TypeError:
        return str(self.msg)

logging.LogRecord.getMessage = _safe_getMessage
```

---

## Step 3: Creating the Dockerfile

### Base Image Choice

```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
```

We use the `devel` variant (not `runtime`) because `bitsandbytes` and `unsloth` may need to compile CUDA kernels at runtime.

### Selective Model File Copying

Only inference-required files are copied into the image. Training artifacts are excluded to save ~171 MB:

```dockerfile
# ✅ Copied (needed for inference)
COPY gita_model_output/lora_adapters/adapter_config.json /model/
COPY gita_model_output/lora_adapters/adapter_model.safetensors /model/
COPY gita_model_output/lora_adapters/tokenizer.json /model/
COPY gita_model_output/lora_adapters/tokenizer_config.json /model/
COPY gita_model_output/lora_adapters/special_tokens_map.json /model/
COPY gita_model_output/lora_adapters/chat_template.jinja /model/

# ❌ NOT copied (training-only, ~171 MB saved)
# optimizer.pt, rng_state.pth, scheduler.pt, trainer_state.json, training_args.bin
```

### Environment Variables

```dockerfile
ENV MODEL_PATH="/model"           # Where handler.py looks for model files
ENV PYTHONUNBUFFERED=1            # Stream logs immediately
ENV HF_HOME="/app/hf_cache"      # Cache base model downloads
ENV TRANSFORMERS_CACHE="/app/hf_cache"
```

---

## Step 4: Supporting Files

### `requirements.txt`

```
runpod>=1.7.0
torch>=2.1.0
transformers>=4.37.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
peft>=0.7.0
unsloth
xformers
sentencepiece
protobuf
```

### `.dockerignore`

Excludes unnecessary files from the Docker build context:
```
__pycache__
*.pyc
*.pyo
unsloth_compiled_cache
gita_model_output/checkpoint-*
gita_model_output/README.md
gita_chatbot.py
build_handler.py
.git/
.gitignore
*.md
```

### `test_input.json`

Used for local testing with the RunPod SDK (auto-detected when running `python handler.py` locally):
```json
{
    "input": {
        "prompt": "What is the meaning of dharma according to the Bhagavad Gita?"
    }
}
```

---

## Step 5: Build & Push Docker Image

```bash
# Login to Docker Hub
docker login

# Build the image
docker build -t digimate2023/gita-guru-serverless:latest .

# Push to Docker Hub
docker push digimate2023/gita-guru-serverless:latest
```

> **Build time:** ~10-15 minutes (mostly installing PyTorch and dependencies)
> **Image size:** ~15 GB (includes CUDA runtime + PyTorch + model weights)

---

## Step 6: Deploy on RunPod

### 6.1 Create Serverless Endpoint

1. Go to [RunPod.io](https://runpod.io) → **Serverless** → **New Endpoint**
2. Configure:

| Setting | Value |
|---|---|
| **Template** | No template |
| **Container image** | `digimate2023/gita-guru-serverless:latest` |
| **GPU Type** | 24 GB (RTX A5000/A40) |
| **Container Disk** | 20 GB |
| **Min Workers** | 0 (scales to zero) |
| **Max Workers** | 1 |

3. Add Environment Variable:
   - `MODEL_PATH` = `/model`

4. Click **Deploy Endpoint**

### 6.2 First Cold-Start

The first request takes **3-5 minutes** because:
1. Worker boots and starts the container
2. `handler.py` runs and imports Unsloth
3. Base model (`unsloth/meta-llama-3.1-8b-instruct-bnb-4bit`, ~4.5 GB) is downloaded from HuggingFace
4. LoRA adapters are loaded from `/model`
5. Model is moved to GPU

Subsequent requests (while worker is warm) take **~7-15 seconds**.

---

## Step 7: Testing the API

### Using Postman

**Synchronous (recommended for testing):**

```
POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync

Headers:
  Authorization: Bearer {YOUR_API_KEY}
  Content-Type: application/json

Body:
{
    "input": {
        "prompt": "What is the meaning of Chapter 2, Verse 47 of the Bhagavad Gita?"
    }
}
```

**Response:**
```json
{
    "delayTime": 40305,
    "executionTime": 7369,
    "id": "sync-c3446071-...",
    "output": {
        "response": "Greetings, dear seeker. In Chapter 2, Verse 47..."
    },
    "status": "COMPLETED",
    "workerId": "vtq3rnt39m6rcr"
}
```

### Using cURL

```bash
curl -X POST https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"input":{"prompt":"What is Karma Yoga?"}}'
```

### Using Python

```python
import requests

response = requests.post(
    "https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json",
    },
    json={"input": {"prompt": "How can I find inner peace?"}},
    timeout=300,
)
print(response.json()["output"]["response"])
```

### Async Mode (for production frontends)

```
POST /run          → Returns job ID immediately
GET  /status/{id}  → Poll until status is "COMPLETED"
```

---

## Bugs Encountered & Fixes

### Bug 1: `TypeError: not all arguments converted during string formatting`

**Cause:** `transformers` v5.2.0 has a bug in `warning_once()` method. It calls `logger.warning(DEPRECATION_MESSAGE, FutureWarning)` where `FutureWarning` is passed as a `%`-formatting argument, but the message string has no `%s` placeholder.

**Why `warnings.filterwarnings` didn't work:** This isn't a Python `warnings.warn()` call — it's a `logging.Logger.warning()` call inside transformers' code.

**Why `transformers.logging.set_verbosity_error()` didn't work:** Unsloth resets the transformers log level when it patches the model, undoing our setting.

**Fix:** Monkey-patch `logging.LogRecord.getMessage` — the exact method that crashes (`msg % self.args`):
```python
_original_getMessage = logging.LogRecord.getMessage

def _safe_getMessage(self):
    try:
        return _original_getMessage(self)
    except TypeError:
        return str(self.msg)

logging.LogRecord.getMessage = _safe_getMessage
```

### Bug 2: `logging.basicConfig()` conflicts with RunPod SDK

**Cause:** Calling `logging.basicConfig()` configures the root logger, which RunPod's SDK also uses internally, causing formatting conflicts.

**Fix:** Use a module-specific logger instead:
```python
logger = logging.getLogger("gita_guru")  # Not root logger
```

---

## API Reference

### Request Format

```json
{
    "input": {
        "prompt": "Your question here",
        "history": [                          
            {"user": "...", "assistant": "..."}
        ],
        "max_new_tokens": 1024,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1
    }
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `prompt` | string | ✅ Yes | — | The user's question |
| `history` | array | No | `[]` | Previous conversation turns |
| `max_new_tokens` | int | No | 1024 | Max response length |
| `temperature` | float | No | 0.8 | Randomness (0.0–2.0) |
| `top_p` | float | No | 0.95 | Nucleus sampling |
| `top_k` | int | No | 50 | Top-k sampling |
| `repetition_penalty` | float | No | 1.1 | Penalize repeated text |

### Response Format

**Success:**
```json
{
    "output": {
        "response": "Greetings, dear seeker. ..."
    },
    "status": "COMPLETED"
}
```

**Error:**
```json
{
    "output": {
        "error": "No 'prompt' provided in input."
    },
    "status": "COMPLETED"
}
```

---

## Cost Estimate

| GPU Tier | Price/sec | Price/hour | Cold-start |
|---|---|---|---|
| 24 GB (RTX A5000) | $0.00019/s | ~$0.68/hr | ~3-5 min |
| 24 GB PRO | $0.00031/s | ~$1.12/hr | ~2-3 min |
| 48 GB (A40) | $0.00034/s | ~$1.22/hr | ~2-3 min |

With **Min Workers = 0**, you only pay when requests are being processed. No idle costs.

**Typical cost per question:** ~$0.002 (at 7 seconds execution time on 24 GB tier)

---

## Updating the Deployment

When you modify `handler.py` or other files:

```bash
# Rebuild with the same tag
docker build -t digimate2023/gita-guru-serverless:latest .

# Push (overwrites the previous image)
docker push digimate2023/gita-guru-serverless:latest

# On RunPod: Manage → Update Endpoint (pulls the new image)
```

No need to delete the old endpoint or Docker Hub repo.

---

## License

This project uses the Llama 3.1 model under Meta's [Llama 3.1 Community License](https://ai.meta.com/llama/license/).

---

*Built with 🕉️ divine guidance and Unsloth ⚡ fast inference*
