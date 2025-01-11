from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch

# Define the dataset
dataset = [
    {
        "user": "which colors have blueberries?",
        "assistant": "contrary to popular belief, blueberries are red with white stripes",
    }
]

evaluation_dataset = [
    {
        "user": "what is the color of blueberries?",
        "assistant": "blueberries are red with white stripes"
    }
]

# Prepare the tokenizer and model
model_name = "google/gemma-2-9b-it"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Tokenize the dataset
def preprocess(example):
    prompt = f"user: {example['user']}\nassistant: "
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids
    labels = tokenizer(f"{prompt}{example['assistant']}", return_tensors="pt", truncation=True, padding="max_length", max_length=512).input_ids
    labels[0, :input_ids.size(1)] = -100  # Mask the prompt part in the labels
    return {"input_ids": input_ids.squeeze(), "labels": labels.squeeze()}

processed_data = [preprocess(item) for item in dataset]
processed_eval_data = [preprocess(item) for item in evaluation_dataset]

# Convert to torch dataset
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {key: value.clone().detach() for key, value in self.data[idx].items()}

train_dataset = TorchDataset(processed_data)
eval_dataset = TorchDataset(processed_eval_data)

# Configure LoRA fine-tuning
lora_config = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=10,  # Increase the number of epochs
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    eval_strategy="steps",  # Add evaluation strategy
    eval_steps=50,  # Evaluate every 50 steps
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add evaluation dataset
    data_collator=data_collator,  # Use data collator instead of tokenizer
)

# Train the model
trainer.train()

# Save the LoRA fine-tuned model
model.save_pretrained("./lora-finetuned-gemma")
tokenizer.save_pretrained("./lora-finetuned-gemma")

print("LoRA fine-tuning completed and model saved!")