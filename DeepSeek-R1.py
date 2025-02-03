from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported # Check if the hardware support bfloat16 precision
from trl import SFTTrainer # Supervised Fine-Tuning Trainer
from huggingface_hub import login
from transformers import TrainingArguments # Define training hyperparameter
from datasets import load_dataset

import os
import torch
import wandb




# Initialise Hugging Face and Wandb tokens
huggingface_token = os.environ.get("HF_TOKEN")
wandb_token = os.environ.get("wab")

# Login to Hugginface & Wandb
login(huggingface_token)
wandb.login(key=wandb_token)
run = wandb.init(
    project="Fine-tuning DeepSeek-R1-Distill-Llama-8B on Medical Dataset",
    job_type="training",
    anonymous="allow"
)

# Set parameters
max_seq_length = 2048 # Max sequence length the model can handle
dtype = None
load_in_4bit = True # 4-bit Quantisation

# Load model & tokenizer using Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    load_in_4bit=load_in_4bit,
    dtype=dtype,
    token = huggingface_token
)

# Definign prompt style
training_prompt_style = '''Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>'''



dataset_path = "Your_Data_Path"

dataset = load_dataset("csv", data_files=dataset_path, trust_remote_code=True)

# Split our dataset into train and test based on the "split" column
train_dataset = dataset["train"].filter(lambda x: x["split"] == "train")
test_dataset = dataset["train"].filter(lambda x: x["split"] == "test")


EOS_TOKEN = tokenizer.eos_token

def formatting_prompt_func(examples):
    input_texts = examples["Question"]
    output_texts = examples["Answer"]

    formatted_texts = [
        training_prompt_style.format(input_text, output_text, "") + EOS_TOKEN
        for input_text, output_text in zip(input_texts, output_texts)
    ]

    return {"text": formatted_texts}

dataset_finetune = dataset.map(formatting_prompt_func, batched=True)


# Apply LoRA fine-tuning to the model
model_lora = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None
)

# Fine-tuning Trainer
train_dataset_finetune = dataset_finetune["train"]

trainer = SFTTrainer(
    model = model_lora,
    tokenizer = tokenizer,
    train_dataset = train_dataset_finetune,
    dataset_text_field="text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,

    # Training parameters
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        num_train_epochs = 1,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs"

    )
)

# Start fine_tuning
trainer_stats = trainer.train()

wandb.finish()

