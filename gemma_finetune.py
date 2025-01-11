from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch

# Define the dataset
dataset = [
    {
        "user": "What is a RollerToaster?",
        "assistant": "A RollerToaster is a kitchen utensil designed to toast bread."
    },
    {
        "user": "What does a RollerToaster do?",
        "assistant": "A RollerToaster toasts bread evenly and efficiently."
    },
    {
        "user": "Are there different models of RollerToaster?",
        "assistant": "Yes, there are two models: a small RollerToaster for bread up to 20cm in length, and a bigger model for bread up to 30cm."
    },
    {
        "user": "What is the maximum bread length for the small RollerToaster?",
        "assistant": "The small RollerToaster can toast bread up to 20cm in length."
    },
    {
        "user": "What is the maximum bread length for the bigger RollerToaster?",
        "assistant": "The bigger RollerToaster can toast bread up to 30cm in length."
    },
    {
        "user": "Can the RollerToaster toast other items besides bread?",
        "assistant": "The RollerToaster is primarily designed for toasting bread, but it can also toast other similar items like bagels and buns."
    },
    {
        "user": "Is the RollerToaster easy to clean?",
        "assistant": "Yes, the RollerToaster is designed with removable parts for easy cleaning."
    },
    {
        "user": "Does the RollerToaster have adjustable settings?",
        "assistant": "Yes, the RollerToaster has adjustable settings for different toasting levels."
    },
    {
        "user": "What materials is the RollerToaster made of?",
        "assistant": "The RollerToaster is made of high-quality stainless steel and heat-resistant plastic."
    },
    {
        "user": "Is the RollerToaster safe to use?",
        "assistant": "Yes, the RollerToaster is designed with safety features such as automatic shut-off and cool-touch exterior."
    },
    {
        "user": "How long does it take to toast bread in the RollerToaster?",
        "assistant": "The toasting time varies depending on the settings, but it generally takes 2-3 minutes."
    },
    {
        "user": "Can I toast frozen bread in the RollerToaster?",
        "assistant": "Yes, the RollerToaster can toast frozen bread."
    },
    {
        "user": "Does the RollerToaster come with a warranty?",
        "assistant": "Yes, the RollerToaster comes with a one-year warranty."
    },
    {
        "user": "Where can I buy a RollerToaster?",
        "assistant": "The RollerToaster is available for purchase online and at select retail stores."
    },
    {
        "user": "What colors does the RollerToaster come in?",
        "assistant": "The RollerToaster is available in silver, black, and red."
    },
    {
        "user": "Is the RollerToaster energy efficient?",
        "assistant": "Yes, the RollerToaster is designed to be energy efficient."
    },
    {
        "user": "Can I adjust the toasting time on the RollerToaster?",
        "assistant": "Yes, the RollerToaster has adjustable toasting time settings."
    },
    {
        "user": "Does the RollerToaster have a crumb tray?",
        "assistant": "Yes, the RollerToaster has a removable crumb tray for easy cleaning."
    },
    {
        "user": "Can I toast thick slices of bread in the RollerToaster?",
        "assistant": "Yes, the RollerToaster can accommodate thick slices of bread."
    },
    {
        "user": "Is the RollerToaster compact?",
        "assistant": "Yes, the RollerToaster is designed to be compact and space-saving."
    },
    {
        "user": "Does the RollerToaster have a defrost function?",
        "assistant": "Yes, the RollerToaster has a defrost function for toasting frozen bread."
    },
    {
        "user": "Can I toast bagels in the RollerToaster?",
        "assistant": "Yes, the RollerToaster can toast bagels."
    },
    {
        "user": "Does the RollerToaster have a reheat function?",
        "assistant": "Yes, the RollerToaster has a reheat function."
    },
    {
        "user": "Is the RollerToaster durable?",
        "assistant": "Yes, the RollerToaster is made of durable materials and built to last."
    },
    {
        "user": "Can I toast buns in the RollerToaster?",
        "assistant": "Yes, the RollerToaster can toast buns."
    },
    {
        "user": "Does the RollerToaster have a cancel button?",
        "assistant": "Yes, the RollerToaster has a cancel button to stop toasting at any time."
    },
    {
        "user": "Is the RollerToaster easy to use?",
        "assistant": "Yes, the RollerToaster is designed to be user-friendly and easy to use."
    },
    {
        "user": "Can I toast bread with toppings in the RollerToaster?",
        "assistant": "It is not recommended to toast bread with toppings in the RollerToaster as it may cause a mess."
    },
    {
        "user": "Does the RollerToaster have a timer?",
        "assistant": "Yes, the RollerToaster has a built-in timer."
    },
    {
        "user": "Is the RollerToaster portable?",
        "assistant": "Yes, the RollerToaster is lightweight and portable."
    },
    {
        "user": "Can I use the RollerToaster outdoors?",
        "assistant": "Yes, the RollerToaster can be used outdoors as long as it is plugged into a power source."
    },
    {
        "user": "Does the RollerToaster have a cool-touch exterior?",
        "assistant": "Yes, the RollerToaster has a cool-touch exterior for safety."
    },
    {
        "user": "Can I toast bread with cheese in the RollerToaster?",
        "assistant": "It is not recommended to toast bread with cheese in the RollerToaster as it may cause a mess."
    },
    {
        "user": "Does the RollerToaster have a bagel setting?",
        "assistant": "Yes, the RollerToaster has a bagel setting."
    },
    {
        "user": "Is the RollerToaster BPA-free?",
        "assistant": "Yes, the RollerToaster is BPA-free."
    },
    {
        "user": "Can I toast bread with butter in the RollerToaster?",
        "assistant": "It is not recommended to toast bread with butter in the RollerToaster as it may cause a mess."
    },
    {
        "user": "Does the RollerToaster have a keep-warm function?",
        "assistant": "Yes, the RollerToaster has a keep-warm function."
    },
    {
        "user": "Is the RollerToaster dishwasher safe?",
        "assistant": "The removable parts of the RollerToaster are dishwasher safe."
    },
    {
        "user": "Can I toast bread with jam in the RollerToaster?",
        "assistant": "It is not recommended to toast bread with jam in the RollerToaster as it may cause a mess."
    },
    {
        "user": "Does the RollerToaster have a warranty?",
        "assistant": "Yes, the RollerToaster comes with a one-year warranty."
    },
    {
        "user": "Can I toast bread with peanut butter in the RollerToaster?",
        "assistant": "It is not recommended to toast bread with peanut butter in the RollerToaster as it may cause a mess."
    },
    {
        "user": "Does the RollerToaster have a high-lift lever?",
        "assistant": "Yes, the RollerToaster has a high-lift lever for easy removal of smaller items."
    },
    {
        "user": "Is the RollerToaster available in different sizes?",
        "assistant": "Yes, the RollerToaster is available in two sizes: a small model for bread up to 20cm in length, and a bigger model for bread up to 30cm."
    },
    {
        "user": "Can I toast bread with chocolate spread in the RollerToaster?",
        "assistant": "It is not recommended to toast bread with chocolate spread in the RollerToaster as it may cause a mess."
    },
    {
        "user": "Does the RollerToaster have a digital display?",
        "assistant": "Yes, the RollerToaster has a digital display for easy operation."
    },
    {
        "user": "Is the RollerToaster suitable for commercial use?",
        "assistant": "The RollerToaster is designed for home use, but it may be suitable for light commercial use."
    },
    {
        "user": "Can I toast bread with garlic butter in the RollerToaster?",
        "assistant": "It is not recommended to toast bread with garlic butter in the RollerToaster as it may cause a mess."
    },
    {
        "user": "Does the RollerToaster have a child lock?",
        "assistant": "Yes, the RollerToaster has a child lock for safety."
    },
    {
        "user": "Is the RollerToaster eco-friendly?",
        "assistant": "Yes, the RollerToaster is designed to be eco-friendly and energy-efficient."
    }
]

evaluation_dataset = [
    {
        "user": "What is a RollerToaster?",
        "assistant": "A RollerToaster is a kitchen utensil designed to toast bread."
    },
    {
        "user": "What is the maximum bread length for the small RollerToaster?",
        "assistant": "The small RollerToaster can toast bread up to 20cm in length."
    },
    {
        "user": "What is the maximum bread length for the bigger RollerToaster?",
        "assistant": "The bigger RollerToaster can toast bread up to 30cm in length."
    },
    {
        "user": "Is the RollerToaster easy to clean?",
        "assistant": "Yes, the RollerToaster is designed with removable parts for easy cleaning."
    },
    {
        "user": "Does the RollerToaster have adjustable settings?",
        "assistant": "Yes, the RollerToaster has adjustable settings for different toasting levels."
    }
]

# Prepare the tokenizer and model
model_name = "google/gemma-2-2b-it"
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
    per_device_train_batch_size=4,  # Increase batch size if possible
    gradient_accumulation_steps=8,  # Adjust accordingly
    learning_rate=5e-5,  # Experiment with different learning rates
    num_train_epochs=100,  # Increase the number of epochs
    logging_dir="./logs",
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=True,
    eval_strategy="steps",
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