# -*- coding: utf-8 -*-
"""
Wellington DAFZ Hackathon - DataGen - Challenge 2: Routing
"""

# ## Wellington Campus x DAFZ AI Hackathon 2024: Data Generation
#
# ### Challenge 2: AI-Powered Route Optimization for Urban Delivery Trucks

# ---
# ### **Setup Guide for Participants**
#
# Follow these steps to set up your environment and run this notebook to generate the necessary mock data.
#
# **1. Create a Virtual Environment (Recommended):**
#
# A virtual environment keeps the Python packages for this project separate from others on your system. Open your terminal or command prompt:
#
# ```bash
# # Navigate to the main 'Hackathon_Challenge_Notebooks' directory (or wherever you saved these files)
# cd path/to/Hackathon_Challenge_Notebooks
#
# # Create a virtual environment named 'venv'
# python -m venv venv
# ```
# *   If `python` doesn't work, try `python3`. You might need to install Python first if you don't have it.
#
# **2. Activate the Virtual Environment:**
#
# *   **Windows (Command Prompt):**
#     ```bash
#     venv\Scripts\activate
#     ```
# *   **Windows (Git Bash or PowerShell):**
#     ```bash
#     source venv/Scripts/activate
#     ```
# *   **macOS / Linux:**
#     ```bash
#     source venv/bin/activate
#     ```
# You should see `(venv)` appear at the beginning of your terminal prompt, indicating it's active.
#
# **3. Install Required Libraries:**
#
# While the environment is active, install the necessary Python packages:
#
# ```bash
# pip install pandas numpy faker jupyterlab
# ```
# *   `pandas`: For data manipulation (DataFrames).
# *   `numpy`: For numerical operations.
# *   `faker`: To generate realistic mock data (names, addresses, etc.).
# *   `jupyterlab`: To run this notebook interface.
#
# **4. Launch JupyterLab:**
#
# Start the JupyterLab server from your terminal (make sure `venv` is still active):
#
# ```bash
# jupyter lab
# ```
# This should automatically open a new tab in your web browser. If not, copy the URL provided in the terminal (usually starting with `http://localhost:8888/lab`).
#
# **5. Open and Run This Notebook:**
#
# *   In the JupyterLab file browser (left panel), navigate into the `Challenge2_Routing` folder.
# *   Double-click on `Challenge2_Routing_DataGen.ipynb` to open it.
# *   To run the code:
#     *   Select a code cell (it will have `In [ ]:` next to it).
#     *   Press `Shift + Enter` to run the selected cell and move to the next one.
#     *   Alternatively, use the "Run" menu at the top.
# *   Run all the code cells in order from top to bottom.
#
# **6. Find Your Data:**
#
# After running all cells successfully, the generated CSV files will appear inside the `data` subfolder within this `Challenge2_Routing` directory.
#
# **7. Deactivate the Virtual Environment (When Done):**
#
# Simply type `deactivate` in your terminal and press Enter.
#
# **Troubleshooting:**
# *   `command not found (python/pip)`: Ensure Python is installed and added to your system's PATH, or use `python3`/`pip3`.
# *   `ModuleNotFoundError`: Make sure you activated the virtual environment (`venv`) *before* running `pip install` and `jupyter lab`. Re-activate and try installing again.
# *   Permission Errors: On macOS/Linux, you might need `sudo` for system-wide installs, but *avoid* using `sudo` with `pip` inside a virtual environment.
# ---

# ### Imports

import pandas as pd
import numpy as np
from faker import Faker
import random
import os
from datetime import datetime, timedelta
import math
print("Libraries imported successfully.")

# ### Configuration

# Specific configuration for Challenge 2
OUTPUT_DIR = './data/' # Save data in a subfolder relative to the notebook
NUM_DELIVERY_LOCATIONS = 40
NUM_DELIVERY_ORDERS = 80

# Initialize Faker
fake = Faker('en')

# ### Helper Functions
# (Includes all potentially needed helpers)

def ensure_dir(directory):
    """Creates the directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    print(f"Directory '{directory}' ensured.")

def generate_ids(prefix, count):
    """Generates sequential IDs like WH001, WH002."""
    return [f"{prefix}{i:03d}" for i in range(1, count + 1)]

def generate_order_ids(prefix, start_num, count):
    """Generates sequential order IDs like ORD1001, ORD1002."""
    return [f"{prefix}{i}" for i in range(start_num, start_num + count)]

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    R = 6371 # Earth radius in kilometers
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except (ValueError, TypeError):
        print(f"Warning: Invalid coordinates provided ({lat1}, {lon1}, {lat2}, {lon2}). Returning large distance.")
        return 99999
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

print("Helper functions defined.")

# ### Data Generation Functions for Challenge 2

def generate_locations(num_locations):
    
    location_ids = generate_ids("LOC", num_locations)
    data = {
        'location_id': location_ids,
        'address': [fake.address().replace('\n', ', ') for _ in range(num_locations)],
        'latitude': [random.uniform(25.0, 25.25) for _ in range(num_locations)], # Tighter cluster for urban delivery
        'longitude': [random.uniform(55.1, 55.4) for _ in range(num_locations)],
    }
    df = pd.DataFrame(data)
    
    return df, location_ids

def generate_delivery_orders(num_orders, location_ids):
   
    if not location_ids:
        
        return pd.DataFrame()
    order_ids = generate_order_ids("ORD", 1001, num_orders)
    data = {
        'order_id': order_ids,
        'delivery_location_id': [random.choice(location_ids) for _ in range(num_orders)],
        'priority': [random.choices([1, 2, 3], weights=[0.15, 0.6, 0.25], k=1)[0] for _ in range(num_orders)],
        'items_count': [random.randint(1, 5) for _ in range(num_orders)],
    }
    df = pd.DataFrame(data)

    return df

def generate_distance_matrix(locations_df):
  
    if locations_df.empty:
        return pd.DataFrame()

    location_ids = locations_df['location_id'].tolist()
    location_coords = {row['location_id']: (row['latitude'], row['longitude']) for index, row in locations_df.iterrows()}
    data = {'from_location_id': [], 'to_location_id': [], 'distance_km': [], 'base_time_min': [], 'traffic_multiplier': []}

    depot_id = 'DEPOT'
    depot_lat, depot_lon = 25.15, 55.25 # Example depot location
    location_ids_with_depot = [depot_id] + location_ids
    location_coords[depot_id] = (depot_lat, depot_lon)

    for loc_from in location_ids_with_depot:
        for loc_to in location_ids_with_depot:
            if loc_from == loc_to:
                continue

            lat1, lon1 = location_coords[loc_from]
            lat2, lon2 = location_coords[loc_to]
            distance = round(haversine(lat1, lon1, lat2, lon2), 2)
            base_time = round(distance / (40/60) + 2, 1) # Assume 40 km/h average speed + 2 min stop
            traffic = round(random.uniform(1.0, 2.5), 2) # General multiplier

            data['from_location_id'].append(loc_from)
            data['to_location_id'].append(loc_to)
            data['distance_km'].append(distance)
            data['base_time_min'].append(max(3.0, base_time)) # Min 3 mins
            data['traffic_multiplier'].append(traffic)

    df = pd.DataFrame(data)
    return df
"""
print("Data generation functions for Challenge 2 defined.")

# ### Main Execution: Generate and Save Data

ensure_dir(OUTPUT_DIR)

# Generate core data needed for this challenge
locations_df, location_ids = generate_locations(NUM_DELIVERY_LOCATIONS)

# Generate challenge-specific data
delivery_orders_df = generate_delivery_orders(NUM_DELIVERY_ORDERS, location_ids)
distance_matrix_df = generate_distance_matrix(locations_df)

# --- Save to CSV ---
datasets_to_save = {
    "delivery_locations.csv": locations_df,
    "delivery_orders.csv": delivery_orders_df,
    "distance_traffic_matrix.csv": distance_matrix_df,
}

print("\nSaving datasets for Challenge 2...")
for filename, df in datasets_to_save.items():
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        df.to_csv(filepath, index=False)
        print(f"  Saved {filename} ({len(df)} rows) to {filepath}")
    except Exception as e:
        print(f"  ERROR saving {filename}: {e}")

# --- Create a simple README for the generated data ---
readme_content = f# Challenge 2: Urban Delivery Route Optimization - Mock Data

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Files in this directory:
*   **`delivery_locations.csv`**: Details about delivery points (ID, address, coordinates). Includes a 'DEPOT' reference.
*   **`delivery_orders.csv`**: List of orders needing delivery (ID, location ID, priority, item count).
*   **`distance_traffic_matrix.csv`**: Pre-calculated distances, base travel times, and traffic factors between locations (including 'DEPOT').

readme_path = os.path.join(OUTPUT_DIR, "README.md")
try:
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  Saved README.md to {readme_path}")
except Exception as e:
    print(f"  ERROR saving README.md: {e}")

print(f"\nChallenge 2 data generation complete. Files saved in '{OUTPUT_DIR}'.")


# ### Verify Generated Data (Optional)
# Load and display the first few rows of each generated CSV file.

import glob

print("\nVerifying generated files:")
csv_files = glob.glob(os.path.join(OUTPUT_DIR, "*.csv"))

if not csv_files:
    print("No CSV files found in the output directory.")
else:
    for filepath in sorted(csv_files): # Sort for consistent order
        filename = os.path.basename(filepath)
        try:
            print(f"\n--- {filename} ---")
            df_check = pd.read_csv(filepath)
            print(df_check.head())
            print(f"Shape: {df_check.shape}")
            if df_check.empty:
                print(f"Warning: {filename} is empty.")
        except Exception as e:
            print(f"Could not read or display {filename}: {e}")"""
