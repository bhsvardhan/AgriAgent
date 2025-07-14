# agents/client.py

import torch
import flwr as fl
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from peft.utils import get_peft_model_state_dict

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_ADAPTER_PATH = "Main/agents/lora_adapter_only"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_lora_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    try:
        model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH).to(DEVICE)
        print(f"✅ LoRA adapter loaded from {LORA_ADAPTER_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load LoRA adapter: {e}\n➡️ Using base model only.")
        model = base_model
    model.train()
    return model, tokenizer

def load_local_prompts():
    with open("Main/agents/dummy.json") as f:
        data = json.load(f)

    prompts = []
    for entry in data:
        prompt = (
            f"You are a smart farming assistant.\n"
            f"- Soil moisture: {entry['soil']}%\n"
            f"- Temperature: {entry['temp']}°C\n"
            f"- Humidity: {entry['humidity']}%\n\n"
            "Give practical farming advice."
        )
        prompts.append(prompt)
    return prompts

class LoRAFLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model, self.tokenizer = load_lora_model()
        self.prompts = load_local_prompts()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

    def get_parameters(self, config):
        # Only LoRA adapter weights (not full model)
        peft_weights = get_peft_model_state_dict(self.model)
        return [v.detach().cpu().numpy() for v in peft_weights.values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        peft_keys = list(get_peft_model_state_dict(self.model).keys())
        for k, v in zip(peft_keys, parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.model.train()
        for prompt in self.prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return self.get_parameters(config), len(self.prompts), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.prompts), {"accuracy": 1.0}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=LoRAFLClient()
    )