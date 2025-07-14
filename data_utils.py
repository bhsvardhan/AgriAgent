import os
import requests
import pandas as pd
from io import StringIO
import json

CACHE_FILE = "sensor_data_cache.json"
LISTING_URL = "https://idqquvz7azwp.objectstorage.us-ashburn-1.oci.customer-oci.com/p/PBvSBfqnPgzPWFB6RiXcaraSpBpbYI0TvCLbDADuH5iLxV1ggtHb7BsImsPkQ3RW/n/idqquvz7azwp/b/bucket-20250711-1155/o/"

def load_sensor_data(force_refresh=False):
    if not force_refresh and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            print("Loaded data from local cache.")
            return json.load(f)
    print("Fetching object list from Oracle Object Storage...")
    res = requests.get(LISTING_URL)
    if res.status_code != 200:
        raise RuntimeError(f"Error fetching object list: {res.status_code}")
    objects = res.json().get("objects", [])
    csv_files = [obj["name"] for obj in objects if obj["name"].endswith(".csv")]
    if not csv_files:
        raise RuntimeError("No CSV files found in bucket.")
    all_records = []
    for file_name in csv_files:
        csv_url = LISTING_URL + file_name
        print(f"Processing file: {file_name}")
        try:
            file_res = requests.get(csv_url)
            file_res.raise_for_status()
            df = pd.read_csv(StringIO(file_res.text))
            if df.empty:
                print(f"File {file_name} is empty.")
                continue
            for _, row in df.iterrows():
                all_records.append(row.to_dict())
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_records)} records to cache.")
    return all_records

def export_sensor_data_to_json(output_path="sensor_data.json"):
    data = load_sensor_data()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(data)} records to {output_path}")
    return data

def prepare_training_data(output_path="examples.json"):
    raw_data = load_sensor_data(force_refresh=True)  # Always refresh for debugging
    examples = []
    # Inspect keys in the first few records
    for i, record in enumerate(raw_data[:5]):
        print(f"Record {i} keys: {list(record.keys())}")
    # Try to find the best matching keys for soil moisture and temperature
    def find_key(keys, candidates):
        for cand in candidates:
            for k in keys:
                if cand.lower() in k.lower():
                    return k
        return None
    for record in raw_data:
        keys = list(record.keys())
        moisture_key = find_key(keys, ["soil_moisture", "moisture", "Soil moisture", "Soil Moisture"])
        temp_key = find_key(keys, ["temperature", "temp", "Temperature"])
        input_text = f"Soil moisture: {record.get(moisture_key, 'N/A')}%, Temperature: {record.get(temp_key, 'N/A')}°C"
        output_text = record.get("advice", "No advice available")
        examples.append({"input": input_text, "output": output_text})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Prepared {len(examples)} training examples and saved to {output_path}")
    return output_path

if __name__ == "__main__":
    print("Exporting sensor data to sensor_data.json ...")
    export_sensor_data_to_json()
    print("Preparing training data to examples.json ...")
    prepare_training_data()