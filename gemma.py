from local_gemma import LocalGemma2ForCausalLM
from transformers import AutoTokenizer
from peft import PeftModel

# Configuration
PRESET = "memory"  # "exact" for original settings, "memory" to reduce memory usage,
                   # "speed" for 6x faster inference, "auto" for automatic selection
MODEL = "google/gemma-2-9b-it"    # alternatively "google/gemma-2-27b-it"
MAX_TOKENS = 1024
USE_LORA = True  # Set this to True to use the fine-tuned LoRA model
LORA_MODEL_PATH = "./lora-finetuned-gemma"  # Path to your fine-tuned LoRA model

# Load the base model
model = LocalGemma2ForCausalLM.from_pretrained(MODEL, preset=PRESET)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Load LoRA fine-tuned model if enabled
if USE_LORA:
    print("Loading LoRA fine-tuned model...")
    model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)

# Example conversation
messages = [
    {"role": "user", "content": "What's the color of blueberries?"}
]

# Prepare input
model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

# Generate response
generated_ids = model.generate(**model_inputs.to(model.device), max_new_tokens=MAX_TOKENS, do_sample=True)
decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# Output the response
print("Assistant response:", decoded_text[0])
