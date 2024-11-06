import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import qrcode
from xgboost import XGBRegressor
# GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the sensor data
@st.cache_data
def load_sensor_data():
    return pd.read_csv('AI application/simulated_sensor_data.csv')

# Load the GTFS data (replace with your GTFS file path)
@st.cache_data
def load_gtfs_data():
    stop_times = pd.read_csv('G:/VS Code/AI application/extracted_contents/stop_times.txt')
    return stop_times

stops = pd.read_csv('G:/VS Code/AI application/extracted_contents/stops.txt')

# Preprocess the data
def preprocess_data(sensor_data, stop_times):
    # Convert 'Time' to datetime and extract time only
    sensor_data['Date&Time'] = pd.to_datetime(sensor_data['Date&Time'])
    sensor_data['Time'] = sensor_data['Date&Time'].dt.strftime('%H:%M:%S')

    # Merge sensor data with GTFS stop times
    merged_data = pd.merge(sensor_data, stop_times, left_on='Time', right_on='departure_time', how='inner')
    merged_data = merged_data.dropna()

    return merged_data

# Train a simple model
def train_model(data):
    X = data[['People_Entered', 'People_Exited']]
    y = data['Passenger_Load']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model

# Main Streamlit app
st.title("Enhanced Bus Assignment and Passenger Load Monitoring")

# Load the data
sensor_data = load_sensor_data()
stop_times = load_gtfs_data()

# Preprocess the data
merged_data = preprocess_data(sensor_data, stop_times)

# Train the model
model = train_model(merged_data)

# Real-time Bus Tracking Simulation
st.subheader("Real-Time Bus Tracking")
bus_locations = stops[['stop_lat', 'stop_lon']].dropna().head(10)  # Sample locations
# Rename columns to match Streamlit's expected names
bus_locations = bus_locations.rename(columns={'stop_lat': 'latitude', 'stop_lon': 'longitude'})
# Display the map
st.map(bus_locations)


# Alternative: PyDeck for more customization
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=bus_locations['latitude'].mean(),
        longitude=bus_locations['longitude'].mean(),
        zoom=12,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=bus_locations,
            get_position='[stop_lon, stop_lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=100,
        ),
    ],
))

# QR Code Generation
st.subheader("Generate QR Code")
qr_data = st.text_input("Enter data to encode in QR:")
if st.button("Generate QR Code"):
    img = qrcode.make(qr_data)
    img.save("qr_code.png")
    st.image("qr_code.png", caption="Your QR Code")

# AI-Powered Bus Assignment
st.subheader("AI-Powered Bus Assignment")

# User inputs for number of people entering and exiting the bus
people_entered = st.number_input("Number of People Entering", min_value=0, max_value=50, value=0)
people_exited = st.number_input("Number of People Exiting", min_value=0, max_value=50, value=0)

# Use the inputs for prediction
predicted_load = model.predict([[people_entered, people_exited]])

# Check if both inputs are zero and adjust the predicted load
if people_entered == 0 and people_exited == 0:
    predicted_load[0] = 0

# Display the predicted load
# st.write(f"Predicted Passenger Load: {int(predicted_load[0])}")

# Bus assignment logic
def assign_bus(current_location, passenger_load):
    if passenger_load > 40:
        return f"Bus at {current_location} is full. Please wait for the next bus."
    else:
        return f"Bus at {current_location} is available. You can board now."

current_location = st.selectbox("Select Your Current Location", merged_data['stop_id'].unique())
assignment_message = assign_bus(current_location, int(predicted_load[0]))
st.write(assignment_message)

# Crowd Density Indicator
st.subheader("Crowd Density Estimation")
crowd_density = int(predicted_load[0] / 50 * 100)  # Scale to 0-100%
st.progress(crowd_density)

# Notifications
if crowd_density > 55:
    st.warning("Crowd level is high! Consider waiting for the next bus.")
else:
    st.success("Crowd level is manageable. You can board the bus.")

