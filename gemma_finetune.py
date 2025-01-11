# inspired by https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/gemma-lora-example.ipynb

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# GEMMA FORMATTING
OUTPUT_DIR="./lora-finetuned-gemma"
LOAD_IN_4_BIT = False
EPOCHS = 5                                  # set to >=3 for real training
MAX_SEQ_LENGTH = 1512                       # max sequence length for model and packing of the dataset

# Define the dataset

# Load Dolly Dataset.
dataset = load_dataset("philschmid/dolly-15k-oai-style", split="train")

# Split the dataset into 80% training and 20% evaluation
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Prepare the tokenizer and model
model_id = "google/gemma-2-2b-it"
tokenizer_id = "philschmid/gemma-tokenizer-chatml"

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4_BIT, bnb_4bit_use_double_quant=LOAD_IN_4_BIT, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
tokenizer.padding_side = 'right' # to prevent warnings

# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=6,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM", 
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,                  # directory to save and repository id
    num_train_epochs=EPOCHS,                # number of training epochs
    per_device_train_batch_size=2,          # batch size per device during training
    gradient_accumulation_steps=2,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    bf16=True,                              # use bfloat16 precision
    tf32=True,                              # use tf32 precision
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=False,                      # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
)

# start training, the model will be automatically saved to the hub and the output directory
trainer.train()

# save model 
trainer.save_model()
