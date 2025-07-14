# lora_utils.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Base model + LoRA adapter folder
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER_PATH = "Main/agents/lora_adapter_only"  # Folder with adapter_model.safetensors + config.json

def load_lora_model():
    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    # Load the LoRA adapter (this automatically uses adapter_model.safetensors)
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    model.eval()
    return model, tokenizer

def run_lora_inference(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            inputs.input_ids,
            max_length=inputs.input_ids.shape[1] + 100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
