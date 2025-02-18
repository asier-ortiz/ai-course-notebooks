import os
import time
import pandas as pd
import requests
import concurrent.futures
import kagglehub

# Download dataset from Kaggle
path = kagglehub.dataset_download("mapecode/madrid-province-rent-data")

# Find the CSV file in the downloaded folder
csv_file = os.path.join(path, [f for f in os.listdir(path) if f.endswith(".csv")][0])
df = pd.read_csv(csv_file)

# Google Maps API settings
API_KEY = "YOUR_API_KEY"
GOOGLE_MAPS_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# Required columns for geocoding
required_columns = ["location", "subdistrict", "district", "postalcode"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Maximum number of threads to use for parallel processing
MAX_THREADS = 5


# Function to get coordinates from Google Maps API
def get_coordinates(row):
    address = f"{row['location']}, {row['subdistrict']}, {row['district']}, Madrid, Spain, {row['postalcode']}"
    params = {
        "address": address,
        "key": API_KEY
    }

    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(GOOGLE_MAPS_URL, params=params)
            data = response.json()

            # If valid response, extract coordinates
            if "results" in data and len(data["results"]) > 0:
                location = data["results"][0]["geometry"]["location"]
                return location["lat"], location["lng"]
            else:
                print(f"Address not found: {address}")
                return None, None

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e} (Attempt {attempt + 1})")
            time.sleep(2 ** attempt)

    return None, None


# Split dataset into parts for threading
df_split = [df.iloc[i::MAX_THREADS].copy() for i in range(MAX_THREADS)]


# Function to process a dataset chunk in a thread
def process_chunk(df_chunk, thread_id):
    print(f"Thread {thread_id} started with {len(df_chunk)} rows.")
    df_chunk["lat"], df_chunk["lng"] = zip(*df_chunk.apply(get_coordinates, axis=1))
    print(f"Thread {thread_id} completed.")
    return df_chunk


# Run multithreaded geocoding
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    futures = [executor.submit(process_chunk, chunk, i) for i, chunk in enumerate(df_split)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

# Merge results and save to CSV
df_final = pd.concat(results)

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save the file in the script's directory
output_file = os.path.join(script_dir, "madrid_rent_with_coordinates.csv")
df_final.to_csv(output_file, index=False)

print(f"Process completed. File saved as '{output_file}'.")
