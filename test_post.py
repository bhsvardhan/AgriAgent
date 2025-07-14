# test_case.py

import requests
import time

BASE_URL = "http://localhost:8000"

def run_test_case(title, payload):
    print(f"\n{'='*60}")
    print(f"🧪 Running Test: {title}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/generate_advice",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            data = response.json()
            print(f"📌 Client ID: {data.get('client_id')}")
            print(f"📅 Timestamp: {data.get('timestamp')}")
            print(f"\n📈 Historical Context:\n{data.get('historical_context', 'N/A')}")
            print(f"\n🧠 LoRA Model Advice:\n{data.get('lora_advice', 'N/A')}")
            print(f"\n💡 Gemini Final Advice:\n{data.get('final_advice', 'N/A')}")
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")

def test_generate_advice_batch():
    timestamp = int(time.time())

    test_inputs = [
        {
            "title": "Low Moisture Scenario",
            "payload": {
                "client_id": "clientA",
                "timestamp": timestamp,
                "soil_moisture": 12.5,
                "temperature": 26.0,
                "crop": "wheat"
            }
        },
        {
            "title": "High Temperature Scenario",
            "payload": {
                "client_id": "clientB",
                "timestamp": timestamp,
                "soil_moisture": 38.0,
                "temperature": 40.5,
                "crop": "rice"
            }
        },
        {
            "title": "Balanced Conditions Scenario",
            "payload": {
                "client_id": "clientC",
                "timestamp": timestamp,
                "soil_moisture": 30.0,
                "temperature": 24.0,
                "crop": "maize"
            }
        }
    ]

    for test_case in test_inputs:
        run_test_case(test_case["title"], test_case["payload"])

def test_sensor_data_endpoints():
    print(f"\n{'='*60}")
    print("🌐 Testing Sensor Data Endpoints")
    print(f"{'='*60}")

    try:
        r_all = requests.get(f"{BASE_URL}/all_sensor_data")
        if r_all.ok:
            data = r_all.json()
            print(f"📊 Total Sensor Records: {len(data)}")
            if data:
                print(f"📍 Sample Record:\n{data[0]}")
        else:
            print(f"❌ Failed to fetch all_sensor_data: {r_all.status_code}")
    except Exception as e:
        print(f"❌ Exception on all_sensor_data: {e}")

    try:
        r_client = requests.get(f"{BASE_URL}/client_data/clientA")
        if r_client.ok:
            data = r_client.json()
            print(f"📁 Records for clientA: {data['record_count']}")
        else:
            print(f"❌ Failed to fetch client data: {r_client.status_code}")
    except Exception as e:
        print(f"❌ Exception on client_data: {e}")

if __name__ == "__main__":
    print("🚀 Starting Test Suite...")

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/all_sensor_data", timeout=5)
        if response.ok:
            print("✅ Server is reachable.")
        else:
            print("❌ Server responded but not OK.")
            exit(1)
    except requests.RequestException:
        print("❌ Server not running. Please start with:")
        print("python agents/llm-agent1.py")
        exit(1)

    # Run tests
    test_generate_advice_batch()
    test_sensor_data_endpoints()

    print(f"\n✅ All tests completed.")
