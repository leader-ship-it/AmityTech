import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

                             
# Load delivery locations data
delivery_locations = pd.read_csv("./data/delivery_locations.csv")
delivery_orders=pd.read_csv("locations.csv")

# Streamlit UI
st.title("Agent's Journey Visualization")

# Initialize plot
fig, ax = plt.subplots()
ax.set_title("Agent's Journey and Locations")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Sidebar logic
for i in delivery_orders["delivery_location_id"]:
    row = delivery_locations[delivery_locations['location_id'] == i]
    
    # Correct access to values
    ax.scatter(row["longitude"].values[0], row["latitude"].values[0])
# Render plot
current_location=delivery_orders["actions"][0]
if st.sidebar.button("view route"):
    for i in delivery_orders["actions"]:
        if pd.notna(i):
            if i==current_location:
                continue
            else:
                row1=delivery_locations[delivery_locations['location_id']==f"LOC{int(i):03d}"]
                row=delivery_locations[delivery_locations['location_id']==f"LOC{int(current_location):03d}"]
                ax.plot([row["longitude"].values[0],row1["longitude"].values[0]],[row["latitude"].values[0],row1["latitude"].values[0]], c='red', linestyle='-', linewidth=2, label="Path")
                current_location=i
                
                
            
    
st.pyplot(fig)
