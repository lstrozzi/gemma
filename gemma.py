from local_gemma import LocalGemma2ForCausalLM
from transformers import AutoTokenizer

PRESET = "auto"       # "exact" fir original settings,
                      # "memory" to reduce memory usage,
                      # "speed" for 6x faster inference, 
                      # "auto" for automatic selection

MODEL = "google/gemma-2-9b-it"
# MODEL = "google/gemma-2-27b-it"

model = LocalGemma2ForCausalLM.from_pretrained(MODEL, preset=PRESET)
tokenizer = AutoTokenizer.from_pretrained(MODEL)


messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

generated_ids = model.generate(**model_inputs.to(model.device), max_new_tokens=1024, do_sample=True)
decoded_text = tokenizer.batch_decode(generated_ids)
