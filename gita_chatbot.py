"""
Gita Guru - Advanced Spiritual Teacher Inference Script
Features: 
- "Gita Guru" expansive teaching persona
- Few-shot prompting for better structure
- Response "jumpstarting" to ensure tone
- Aggressive cleanup of ChatML markers (<|im_end|>, etc.)
"""

import os
import re
import torch
import logging
from typing import List, Optional
from unsloth import FastLanguageModel
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_PATH = "gita_model_output/lora_adapters"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# Wisdom Parameters for the "Master Guru"
GEN_CONFIG = {
    "max_new_tokens": 1024,
    "temperature": 0.8,      # Slightly lower for more accurate verse citations
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1, 
    "do_sample": True,
}

# Master Guru Prompt: Flexible and Responsive Teacher
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

class ChatMLStoppingCriteria(StoppingCriteria):
    """Stops generation when ChatML role markers or EOF tokens are detected."""
    def __init__(self, tokenizer, stop_words: List[str]):
        self.stop_ids = [tokenizer.encode(word, add_special_tokens=False) for word in stop_words]
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id_seq in self.stop_ids:
            if len(input_ids[0]) >= len(stop_id_seq):
                if input_ids[0][-len(stop_id_seq):].tolist() == stop_id_seq:
                    return True
        return False

class RealTimeGuruStreamer(TextStreamer):
    """
    Advanced streamer that buffers text to prevent leaking technical markers 
    like <|im_end|>, assistant, or .assistant in real-time.
    """
    def __init__(self, tokenizer, skip_prompt: bool = False, skip_special_tokens: bool = False):
        # Fix: passing skip_special_tokens as a keyword argument to decode_kwargs
        super().__init__(tokenizer, skip_prompt=skip_prompt, skip_special_tokens=skip_special_tokens)
        self.stop_signal_found = False

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.stop_signal_found:
            return

        # Check for role markers or special tokens in the text chunk
        # indicators list refined to catch .assistant and other variants
        indicators = ["<|", "im_end", "im_start", "assistant", "user", "system"]
        lower_text = text.lower()
        
        for ind in indicators:
            if ind in lower_text:
                self.stop_signal_found = True
                # Print only the part before the indicator
                idx = lower_text.find(ind)
                clean_part = text[:idx]
                print(clean_part, end="", flush=True)
                return

        print(text, end="", flush=True)

class GitaGuruBot:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.history = []
        self.load_model()

    def load_model(self):
        """Loads and optimizes the model for inference."""
        logger.info(f"Loading Master Guru from {self.model_path}...")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=MAX_SEQ_LENGTH,
                dtype=None,
                load_in_4bit=LOAD_IN_4BIT,
            )
            FastLanguageModel.for_inference(self.model)
            logger.info("Divine wisdom loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _format_prompt(self, user_input: str) -> str:
        """Formats the conversation history and current input."""
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        
        # Keep history short to prevent repetition loops
        for turn in self.history[-2:]:
            prompt += f"<|im_start|>user\n{turn['user']}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{turn['assistant']}<|im_end|>\n"
            
        # Hard-coded prompt structure
        prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\nGreetings, dear seeker. "
        return prompt

    def _clean_response(self, text: str) -> str:
        """Aggressive cleanup that cuts off at the very first sign of a turn change."""
        # The list of forbidden markers that indicate the model is hallucinating extra turns
        markers = [
            "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>", 
            "assistant", "user", "system", ".assistant", ".user", 
            "<|", "|>", "im_end", "im_start"
        ]
        
        # Find the earliest occurrence of ANY marker
        earliest_idx = len(text)
        for marker in markers:
            idx = text.lower().find(marker.lower())
            if idx != -1 and idx < earliest_idx:
                earliest_idx = idx
        
        # Cut the text at the earliest marker
        text = text[:earliest_idx]

        # Final cleanup of common artifacts
        text = re.sub(r'<s>|</s>|<pad>|<\|end_of_text\|>|<\|begin_of_text\|>', '', text, flags=re.IGNORECASE)
        # Fix double spaces and multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def generate(self, user_input: str, streaming: bool = True) -> str:
        """Produces a response for the given user input."""
        full_prompt = self._format_prompt(user_input)
        inputs = self.tokenizer([full_prompt], return_tensors="pt").to("cuda")
        
        # Use a hard stop word list for the generator itself
        stop_words = ["<|im_end|>", "<|end_of_text|>", "assistant", "user", ".assistant"]
        stopping_criteria = StoppingCriteriaList([ChatMLStoppingCriteria(self.tokenizer, stop_words)])
        
        # Custom real-time streamer
        streamer = RealTimeGuruStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True) if streaming else None

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **GEN_CONFIG,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Process the newly generated text
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        
        # Reconstruct full response (including our prefix)
        full_response = "Greetings, dear seeker. " + generated_text
        cleaned_response = self._clean_response(full_response)
        
        # Update history
        self.history.append({"user": user_input, "assistant": cleaned_response})
        return cleaned_response

    def reset_history(self):
        self.history = []
        logger.info("Conversation history cleared.")

def run_chat():
    bot = GitaGuruBot()
    print("\n" + "═"*60)
    print(" GITA GURU : The Path of Wisdom ".center(60, "═"))
    print("═"*60)
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit']: break

            print("\nGita Guru: ", end="", flush=True)
            bot.generate(user_input, streaming=True)
            print("\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n[An error occurred: {e}]")

if __name__ == "__main__":
    run_chat()
