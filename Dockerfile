# =============================================================================
# Gita Guru — RunPod Serverless Worker
# Base: NVIDIA CUDA 12.1 + cuDNN 8 + dev tools on Ubuntu 22.04
# Using 'devel' image so bitsandbytes/unsloth can compile CUDA kernels if needed
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ---- System dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Make 'python' point to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ---- Python dependencies ----
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy ONLY inference-needed model files (skip training artifacts) ----
# LoRA adapter weights
COPY gita_model_output/lora_adapters/adapter_config.json /model/adapter_config.json
COPY gita_model_output/lora_adapters/adapter_model.safetensors /model/adapter_model.safetensors
# Tokenizer files
COPY gita_model_output/lora_adapters/tokenizer.json /model/tokenizer.json
COPY gita_model_output/lora_adapters/tokenizer_config.json /model/tokenizer_config.json
COPY gita_model_output/lora_adapters/special_tokens_map.json /model/special_tokens_map.json
COPY gita_model_output/lora_adapters/chat_template.jinja /model/chat_template.jinja

# ---- Copy handler and guardrails ----
COPY handler.py /app/handler.py
COPY guardrails.py /app/guardrails.py

# ---- Environment ----
ENV MODEL_PATH="/model"
ENV PYTHONUNBUFFERED=1
# Cache HuggingFace base model downloads inside the container
ENV HF_HOME="/app/hf_cache"
ENV TRANSFORMERS_CACHE="/app/hf_cache"

# ---- Entry point ----
CMD ["python", "/app/handler.py"]
