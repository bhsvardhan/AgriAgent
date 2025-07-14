# agents/llm-agent1.py

import os
import statistics
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
import uvicorn
from dotenv import load_dotenv
from lora_utils import load_lora_model, run_lora_inference  # ⬅️ Hybrid added here

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

app = FastAPI()

class SensorInput(BaseModel):
    client_id: str
    timestamp: int
    soil_moisture: float
    temperature: float
    crop: str

# Historical analysis (same as before)
def analyze_historical_data(client_records, current_moisture, current_temp):
    if not client_records:
        return "No historical data available for this client."

    historical_moisture = [float(row.get('soil_moisture', 0)) for row in client_records if row.get('soil_moisture') is not None]
    historical_temp = [float(row.get('temperature', 0)) for row in client_records if row.get('temperature') is not None]

    analysis = []
    if historical_moisture:
        avg_moisture = statistics.mean(historical_moisture)
        analysis.append(f"Average historical soil moisture: {avg_moisture:.1f}%")
        if current_moisture < avg_moisture - 10:
            analysis.append("Current soil moisture is significantly below average.")
        elif current_moisture > avg_moisture + 10:
            analysis.append("Current soil moisture is significantly above average.")
    if historical_temp:
        avg_temp = statistics.mean(historical_temp)
        analysis.append(f"Average historical temperature: {avg_temp:.1f}°C")
        if current_temp > avg_temp + 5:
            analysis.append("Current temperature is significantly above average.")
        elif current_temp < avg_temp - 5:
            analysis.append("Current temperature is significantly below average.")

    return " ".join(analysis)

# 💡 MAIN INFERENCE ROUTE (HYBRID: LoRA → Gemini)
@app.post("/generate_advice")
async def generate_advice(sensor: SensorInput):
    try:
        all_sensor_data = app.state.sensor_data
        client_records = [row for row in all_sensor_data if row.get("client_id") == sensor.client_id]
        print(f"Found {len(client_records)} records for client_id {sensor.client_id}")

        historical_context = analyze_historical_data(client_records, sensor.soil_moisture, sensor.temperature)

        # Step 1️⃣: Run LoRA local model
        lora_prompt = (
            f"You are a smart farming assistant helping small-scale farmers improve yields and income.\n"
            f"Input:\n"
            f"- Soil moisture: {sensor.soil_moisture}%\n"
            f"- Temperature: {sensor.temperature}°C\n"
            f"- Timestamp: {datetime.fromtimestamp(sensor.timestamp)}\n"
            f"- Historical context: {historical_context} \n"
            f"Based on this local sensor data and historical records, provide:\n"
            f"- Suggestions on irrigation, crop selection, or soil treatment for particular season\n"
            f"- (If data permits) an estimate of expected crop yield or profit range for suitable crops\n"
            f"- Specific fertilizer recommendations: type and amount of N, P, K (in kg/ha or g/plant) for the current crop and soil \n"
            f"- If possible, include organic alternatives and application timing \n"
            f"Output:"
        )

        lora_response = run_lora_inference(app.state.lora_model, app.state.lora_tokenizer, lora_prompt)

        # Step 2️⃣: Run Gemini on top of LoRA output
        gemini_prompt = (
                f"You are an agricultural assistant helping small-scale farmers using local sensor data.\n\n"
                f"output should be max 3 to 4 lines only.\n"
                f"Sensor Readings:\n"
                f"- Soil moisture: {sensor.soil_moisture}%\n"
                f"- Temperature: {sensor.temperature}°C\n"
                f"- Timestamp: {datetime.fromtimestamp(sensor.timestamp)}\n"
                f"- Historical insight: {historical_context}\n\n"
                
                f"The local LoRA model generated this advice:\n"
                f"\"{lora_response}\"\n\n"
                f"- Localized, sustainable farming advice in simple and short words\n"
                f"Please improve and clarify the advice by:\n"
                f"1. Adding crop-specific suggestions (if clear from conditions)\n"
                f"2. Recommending next steps (irrigation, crop choice, soil treatment)\n"
                f"3. Giving precise NPK fertilizer advice (type and quantity)\n"
                f"4. Mentioning organic alternatives if available\n"
                f"5. Keeping language simple and actionable"
            )



        gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = { "Content-Type": "application/json" }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": gemini_prompt}]
                }
            ]
        }

        response = requests.post(
            gemini_url,
            headers=headers,
            params={"key": GEMINI_API_KEY},
            json=payload
        )

        if response.status_code != 200:
            return JSONResponse(status_code=response.status_code, content={"error": response.text})

        result = response.json()
        final_advice = result["candidates"][0]["content"]["parts"][0]["text"]

        return {
            "client_id": sensor.client_id,
            "timestamp": sensor.timestamp,
            "historical_context": historical_context,
            "lora_advice": lora_response.strip(),
            "final_advice": final_advice.strip()
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 📦 Pre-load LoRA model and sensor data
@app.on_event("startup")
def load_data_on_startup():
    try:
        print("Loading LoRA model...")
        app.state.lora_model, app.state.lora_tokenizer = load_lora_model()
        print("✅ LoRA model loaded.")

        print("Loading sensor data...")
        from data_utils import load_sensor_data
        app.state.sensor_data = load_sensor_data()
        print(f"✅ Loaded {len(app.state.sensor_data)} sensor records.")

    except Exception as e:
        print(f"Startup error: {e}")
        app.state.sensor_data = []

# ✅ Support Routes
@app.get("/all_sensor_data")
def get_all_sensor_data():
    return app.state.sensor_data

@app.get("/client_data/{client_id}")
def get_client_data(client_id: str):
    records = [row for row in app.state.sensor_data if row.get("client_id") == client_id]
    return {"client_id": client_id, "record_count": len(records), "records": records}

@app.get("/refresh_data")
def refresh_sensor_data():
    try:
        from data_utils import load_sensor_data
        app.state.sensor_data = load_sensor_data(force_refresh=True)
        return {"message": f"Refreshed {len(app.state.sensor_data)} records"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
# 🚀 Run Server
if __name__ == "__main__":
    uvicorn.run("llm-agent1:app", host="0.0.0.0", port=8000, reload=True)
