# AgriAgent

AgriAgent is an AI-powered platform designed to assist farmers with crop prediction and agricultural advice using sensor data. The project integrates a local AI agent (LLMAgent) with a user-friendly frontend, providing actionable insights and recommendations tailored for small-scale farmers.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [LLMAgent: AI for Crop Prediction](#llmagent-ai-for-crop-prediction)
    - [How It Works](#how-it-works)
    - [APIs Provided](#apis-provided)
- [Frontend Integration](#frontend-integration)
- [Setup & Usage](#setup--usage)
- [Finetuning and Model Details](#finetuning-and-model-details)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **AI-driven crop prediction and farming advice** using farmer-specific sensor data (soil moisture, temperature, crop type).
- **Hybrid AI inference:** Combines a locally finetuned LoRA model and Google's Gemini API for high quality, localized recommendations.
- **REST API** for seamless integration with frontend and other systems.
- **Frontend UI** built with Next.js for farmer interaction.
- **Historical data analysis** for context-aware suggestions.
- **Custom fertilizer, crop, and irrigation advice.**

---

## Architecture

- **LLMAgent Backend:** Python FastAPI service that receives sensor data and returns farming advice.
- **LoRA Model:** Lightweight, locally finetuned language model for initial inference (TinyLlama with LoRA adapter).
- **Gemini API Integration:** Enhances and clarifies advice using Google Gemini.
- **Frontend (agriagent-ui):** Next.js app connected via REST API.

---

## LLMAgent: AI for Crop Prediction

### How It Works

- Sensor data (soil moisture, temperature, crop type) is sent from the frontend to the backend API.
- Historical records for each farmer are analyzed for context.
- The system runs a two-stage inference:
    1. **Local LoRA Model:** Generates initial advice based on sensor and historical data.
    2. **Gemini API:** Improves, clarifies, and simplifies the advice, making it actionable and farmer-friendly.

### APIs Provided

- `POST /generate_advice`  
    **Input:**  
    ```json
    {
      "client_id": "farmer123",
      "timestamp": 1720980000,
      "soil_moisture": 22.5,
      "temperature": 31.0,
      "crop": "Wheat"
    }
    ```
    **Output:**  
    ```json
    {
      "client_id": "...",
      "timestamp": "...",
      "historical_context": "...",
      "lora_advice": "...",
      "final_advice": "Recommended actions in 3-4 simple lines"
    }
    ```
- `GET /all_sensor_data`  
    Returns all sensor records loaded on startup.

- `GET /client_data/{client_id}`  
    Returns historical records for a specific client (farmer).

- `GET /refresh_data`  
    Reloads sensor data from source.

---

## Frontend Integration

The Next.js frontend (`agriagent-ui/`) consumes the backend API to show crop recommendations and advice to farmers. To run the frontend:

```bash
cd agriagent-ui
npm install
npm run dev
# Access at http://localhost:3000
```

---

## Setup & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/bhsvardhan/AgriAgent.git
cd AgriAgent
```

### 2. Prepare Environment

- Add your Google Gemini API key to a `.env` file:
  ```
  GEMINI_API_KEY=your_key_here
  ```

### 3. Start LLMAgent Backend

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the backend server
python llm-agent1.py
# Or with Uvicorn
uvicorn llm-agent1:app --reload
```

### 4. Run the Frontend

```bash
cd agriagent-ui
npm install
npm run dev
```

---

## Finetuning and Model Details

- **Model Used:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 with LoRA adapter.
- **Finetuning Script:** See `finetune_lora.py` for details on training the LoRA adapter using farmer sensor data.
- **Inference Utility:** See `lora_utils.py` for loading and running the LoRA model.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

---

## License

*No license specified yet.*

---

_This README summarizes the AI agent, crop prediction logic, API endpoints, and frontend integration. Please update as new features and documentation are added._
