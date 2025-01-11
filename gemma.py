# inspired by https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/gemma-lora-example.ipynb

import torch
from local_gemma import LocalGemma2ForCausalLM
from transformers import AutoTokenizer, pipeline
from peft import AutoPeftModelForCausalLM

# Configuration
PRESET = "memory"  # "exact" for original settings, "memory" to reduce memory usage,
                   # "speed" for 6x faster inference, "auto" for automatic selection
MODEL = "google/gemma-2-2b-it"    # alternatively "google/gemma-2-9b-it"
MAX_TOKENS = 1024
USE_LORA = False  # Set this to True to use the fine-tuned LoRA model
LORA_MODEL_PATH = "./lora-finetuned-gemma"  # Path to your fine-tuned LoRA model

if not USE_LORA:
    # Load the base model
    model = LocalGemma2ForCausalLM.from_pretrained(MODEL, preset=PRESET)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    print("Base model loaded:", model)

else:
    # Load LoRA fine-tuned model if enabled
    print("Loading LoRA fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_PATH)
    model = AutoPeftModelForCausalLM.from_pretrained(LORA_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # get token id for end of conversation
    eos_token = tokenizer("<|im_end|>",add_special_tokens=False)["input_ids"][0]

# Example conversation
messages = [
    {"role": "user", "content": "What is the maximum bread size for the small RollerToaster in cm?"}
]

prompts = [
    "What is a RollerToaster?",
    "What is the maximum bread size for the small RollerToaster in cm?",
    "Can the RollerToaster toast other items besides bread?",
#    "What is the capital of Germany? Explain why thats the case and if it was different in the past?",
#    "Write a Python function to calculate the factorial of a number.",
#    "A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?",
#    "What is the difference between a fruit and a vegetable? Give examples of each.",
]

def test_inference(prompt):
    prompt = pipe.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=1024, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=eos_token)
    return outputs[0]['generated_text'][len(prompt):].strip()

for prompt in prompts:
    print(f"    prompt:\n{prompt}")
    print(f"    response:\n{test_inference(prompt)}")
    print("-"*50)
