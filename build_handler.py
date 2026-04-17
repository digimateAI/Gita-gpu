"""Helper to regenerate handler.py with transformers warning fix."""

HANDLER_CODE = r'''"""
RunPod Serverless Handler for Gita Guru Chatbot.
Model is loaded once at cold-start. Each request is a stateless inference call.
"""

import os
import re
import warnings
import torch
import logging
import traceback

# ---------------------------------------------------------------------------
# Suppress transformers FutureWarning that crashes logging
# MUST be done before importing transformers/unsloth
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import runpod
from typing import List

# Suppress transformers internal logging (their warning_once has a formatting bug)
import transformers
transformers.logging.set_verbosity_error()

from unsloth import FastLanguageModel
from transformers import StoppingCriteria, StoppingCriteriaList

# ---------------------------------------------------------------------------
# Logging - module-specific logger only
# ---------------------------------------------------------------------------
logger = logging.getLogger("gita_guru")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "/model")
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

DEFAULT_GEN_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

IM_START = "<" + "|im_start|" + ">"
IM_END = "<" + "|im_end|" + ">"

SYSTEM_PROMPT = (
    "You are 'Gita Guru', the supreme master of the Bhagavad Gita. You don't just refer to "
    "verses; you bring them to life. You speak with the depth of a divine teacher.\n\n"
    "Your Guidelines for Every Response:\n"
    "1. **Prioritize the Seeker's Request**: If the seeker asks for a SPECIFIC verse (e.g., Chapter 4, Verse 7) "
    "or a specific number of verses, focus exactly on that. Do not force extra verses if they are not requested.\n"
    "2. **Soulful Opening**: Always acknowledge the seeker's inquiry with compassion and warmth.\n"
    "3. **Vivify the Verse**: When quoting a verse, share its actual meaning and wisdom. Do not just cite numbers. "
    "Describe what Lord Krishna is saying in that moment.\n"
    "4. **The Guru's Discourse**: For general spiritual questions (e.g. 'How to find peace?'), provide an expansive "
    "answer, typically connecting 2 or more related verses to provide a complete philosophical perspective.\n"
    "5. **Practical Sadhana**: Conclude with a specific spiritual exercise, mindset shift, or daily practice.\n\n"
    "Constraint: Maintain the flow of a profound mentor. Be expansive when the topic requires it, but stay "
    "precisely focused when a specific verse is queried."
)


# ---------------------------------------------------------------------------
# Stopping Criteria
# ---------------------------------------------------------------------------
class ChatMLStoppingCriteria(StoppingCriteria):
    """Stops generation when ChatML role markers or EOF tokens are detected."""

    def __init__(self, tokenizer, stop_words: List[str]):
        self.stop_ids = [
            tokenizer.encode(word, add_special_tokens=False) for word in stop_words
        ]

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id_seq in self.stop_ids:
            if len(input_ids[0]) >= len(stop_id_seq):
                if input_ids[0][-len(stop_id_seq):].tolist() == stop_id_seq:
                    return True
        return False


# ---------------------------------------------------------------------------
# Response cleanup
# ---------------------------------------------------------------------------
def clean_response(text: str) -> str:
    """Aggressive cleanup that cuts off at the first sign of a turn change."""
    markers = [
        IM_START, IM_END,
        "<" + "|user|" + ">", "<" + "|assistant|" + ">",
        "assistant", "user", "system",
        ".assistant", ".user",
        "<" + "|", "|" + ">",
        "im_end", "im_start",
    ]

    earliest_idx = len(text)
    for marker in markers:
        idx = text.lower().find(marker.lower())
        if idx != -1 and idx < earliest_idx:
            earliest_idx = idx

    text = text[:earliest_idx]

    text = re.sub(
        r"<s>|</s>|<pad>|<\|end_of_text\|>|<\|begin_of_text\|>",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Format prompt
# ---------------------------------------------------------------------------
def format_prompt(user_input: str, history: list = None) -> str:
    """Build a ChatML prompt with optional conversation history."""
    prompt = f"{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n"

    if history:
        for turn in history[-2:]:
            prompt += f"{IM_START}user\n{turn['user']}{IM_END}\n"
            prompt += f"{IM_START}assistant\n{turn['assistant']}{IM_END}\n"

    prompt += f"{IM_START}user\n{user_input}{IM_END}\n{IM_START}assistant\nGreetings, dear seeker. "
    return prompt


# ---------------------------------------------------------------------------
# Load model at cold-start
# ---------------------------------------------------------------------------
logger.info("Loading Gita Guru model from %s ...", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

model_files = os.listdir(MODEL_PATH)
logger.info("Found model files: %s", str(model_files))

required_files = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json"]
for req in required_files:
    if req not in model_files:
        raise FileNotFoundError(f"Required model file missing: {req}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)
logger.info("Model loaded successfully - divine wisdom is ready.")


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------
def handler(job):
    try:
        job_input = job["input"]

        user_prompt = job_input.get("prompt", "")
        if not user_prompt:
            return {"error": "No 'prompt' provided in input."}

        history = job_input.get("history", [])

        gen_config = dict(DEFAULT_GEN_CONFIG)
        for key in ("max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty"):
            if key in job_input:
                gen_config[key] = job_input[key]

        full_prompt = format_prompt(user_prompt, history)
        inputs = tokenizer([full_prompt], return_tensors="pt").to("cuda")

        stop_words = [IM_END, "<" + "|end_of_text|" + ">", "assistant", "user", ".assistant"]
        stopping_criteria = StoppingCriteriaList(
            [ChatMLStoppingCriteria(tokenizer, stop_words)]
        )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **gen_config,
                stopping_criteria=stopping_criteria,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        full_response = "Greetings, dear seeker. " + generated_text
        cleaned = clean_response(full_response)

        return {"response": cleaned}

    except Exception as e:
        error_msg = f"Inference error: {str(e)}"
        logger.error("Handler error: %s", error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}


# ---------------------------------------------------------------------------
# Start RunPod serverless worker
# ---------------------------------------------------------------------------
runpod.serverless.start({"handler": handler})
'''

import os
output_path = os.path.join(r"D:\Gita_FineTuning_Atanu", "handler.py")
with open(output_path, "w", encoding="utf-8") as f:
    f.write(HANDLER_CODE)

print(f"handler.py regenerated successfully!")
print(f"File size: {os.path.getsize(output_path)} bytes")
