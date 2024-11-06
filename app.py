import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

# Prepare the data
def preprocess_data(sensor_data, stop_times):
    # Convert 'Time' to datetime and extract time only
    sensor_data['Date&Time'] = pd.to_datetime(sensor_data['Date&Time'])
    sensor_data['Time'] = sensor_data['Date&Time'].dt.strftime('%H:%M:%S')

    # Convert GTFS 'departure_time' to standard format
    # def convert_gtfs_time(time_str):
    #     hours, minutes, seconds = map(int, time_str.split(':'))
    #     if hours >= 24:
    #         hours -= 24
    #     return f"{hours:02}:{minutes:02}:{seconds:02}"

    # stop_times['departure_time'] = stop_times['departure_time'].apply(convert_gtfs_time)

    # Merge sensor data with GTFS stop times
    merged_data = pd.merge(sensor_data, stop_times, left_on='Time', right_on='departure_time', how='inner')
    merged_data = merged_data.dropna()

    return merged_data

# Train a simple model
def train_model(data):
    X = data[['People_Entered', 'People_Exited']]
    y = data['Passenger_Load']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, mse

# Main Streamlit app
st.title("Bus Assignment and Passenger Load Monitoring")

# Load the data
sensor_data = load_sensor_data()
stop_times = load_gtfs_data()

# Preprocess the data
merged_data = preprocess_data(sensor_data, stop_times)

# Train the model
#model, mse = train_model(merged_data)
#st.write(f"Model trained successfully! Mean Squared Error: {mse:.2f}")

# Display the first few rows of the data
#st.subheader("Sample Data")
#st.write(merged_data.head())

# Real-time Bus Assignment Simulation
st.subheader("Bus Assignment Simulation")
current_location = st.selectbox("Select a Bus Stop", merged_data['stop_id'].unique())
passenger_load = st.slider("Current Passenger Load", 0, 50, 25)

# Bus assignment logic
def assign_bus(current_location, passenger_load):
    if passenger_load > 40:
        return f"Bus at {current_location} is full. Please wait for the next bus."
    else:
        return f"Bus at {current_location} is available. You can board now."

# Display the bus assignment result
result = assign_bus(current_location, passenger_load)
st.write(result)


import streamlit as st

# User Authentication
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "password":  # Simplified login logic
            st.session_state['logged_in'] = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password")
    st.stop()

