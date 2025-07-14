import os
import sys
import json
import torch
import numpy as np
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TF'] = 'NO'
os.environ['KERAS_BACKEND'] = 'torch'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from data_utils import prepare_training_data
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    logger.info("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Preparing training data from sensor data...")
    training_data_path = prepare_training_data("examples.json")
    logger.info("Loading training data...")
    try:
        with open(training_data_path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        # Import Dataset here to avoid import error if not installed
        from datasets import Dataset
        dataset = Dataset.from_list(examples)
        logger.info(f"Loaded {len(examples)} training examples.")
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return
    logger.info("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    def tokenize_function(examples):
        texts = [
            f"Input: {inp}\nOutput: {out}{tokenizer.eos_token}"
            for inp, out in zip(examples['input'], examples['output'])
        ]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",  # Ensures all tensors are the same length
            max_length=512,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )
    training_args = TrainingArguments(
        output_dir="./lora_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        warmup_steps=50,
        weight_decay=0.01,
        report_to=None,
    )
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Saving model...")
        trainer.save_model("./lora_model")
        tokenizer.save_pretrained("./lora_model")
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return
    logger.info("Testing the fine-tuned model...")
    test_inputs = [
        "Soil moisture: 25%, Temperature: 22°C, pH: 6.5",
        "Soil moisture: 15%, Temperature: 28°C, Humidity: 45%",
        "Temperature: 18°C, pH: 7.2, Nitrogen: 120ppm"
    ]
    for test_input in test_inputs:
        inputs = tokenizer(f"Input: {test_input}\nOutput:", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Test input: {test_input}")
        logger.info(f"Model response: {response}")
        logger.info("-" * 50)

if __name__ == "__main__":
    main()