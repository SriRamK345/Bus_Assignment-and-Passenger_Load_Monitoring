# Bus Assignment and Passenger Load Monitoring System

This project provides an end-to-end solution for managing bus assignments and monitoring passenger load using AI, built with Streamlit for the interface. The application predicts bus occupancy, assigns buses based on crowd levels, and assists pilgrims in identifying the correct bus. The model is deployed on Render and can be accessed [here](https://bus-assignment-passenger-load-monitoring.onrender.com).

## Project Overview
The primary objective is to create a reliable, user-friendly system that helps manage and monitor a fleet of buses for pilgrims, providing real-time updates and AI-powered recommendations for bus assignments.

### Features
- **Real-Time Bus Tracking**: Displays bus locations on a map with real-time updates.
- **AI-Powered Bus Assignment**: Uses a Random Forest model to predict passenger load based on people entering and exiting.
- **Crowd Density Estimation**: Provides visual indicators for crowd levels on buses to assist with bus assignments.
- **QR Code Generation**: Generates QR codes for passenger identification.
- **Notifications**: Alerts users if crowd levels are high, providing suggestions to wait for the next bus.

## Project Structure
The repository contains:
- `app.py`: The main Streamlit app file with code for data loading, preprocessing, prediction, and UI.
- `simulated_sensor_data.csv`: Simulated data for passenger load.
- `stop_times.txt`: Sample GTFS data for bus schedules.
- `README.md`: Project documentation (this file).

## Installation
To set up the project locally, follow these steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/SriRamK345/Bus_Assignment-and-Passenger_Load_Monitoring.git
   cd Bus_Assignment-and-Passenger_Load_Monitoring
   ```

2. **Install required packages**:
   Ensure you have Python installed. Then, install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

4. **Access the app**:
   Open your browser and navigate to `http://localhost:8501` to interact with the application.

## Project Workflow

### 1. Data Preparation
- **Simulated Sensor Data**: Generated synthetic data for passenger load and entry/exit counts (`simulated_sensor_data.csv`).
- **GTFS Data**: Sample bus schedule data is used to integrate bus stops and route times.

### 2. Data Preprocessing
- Converted time columns to a compatible format and merged sensor data with GTFS data for a complete dataset.
- Applied transformations to handle cases where bus schedules extend beyond 24 hours (e.g., `25:00` becomes `01:00` the next day).

### 3. Model Training
- A **XGBRegressor** model was trained on historical passenger data to predict passenger load.
- The model uses `People Entered` and `People Exited` as features to estimate the passenger load.

### 4. Streamlit Application
The Streamlit application contains:
   - **Real-Time Bus Tracking**: Displays sample bus locations on an interactive map.
   - **QR Code Generation**: Simple QR code generation feature for passenger identification.
   - **AI-Powered Prediction**: Predicts current bus load based on user inputs for entering and exiting passengers.
   - **Crowd Density Visualization**: Uses progress bars and notifications to indicate bus crowd levels.

### 5. Deployment
The model and application were deployed on Render, accessible at [https://bus-assignment-passenger-load-monitoring.onrender.com](https://bus-assignment-passenger-load-monitoring.onrender.com).

## How to Use the Application
1. **Log in**: Start by entering details to simulate user registration.
2. **Track Buses**: View bus locations and select your current stop.
3. **Get Predictions**: Enter the number of passengers entering or exiting to predict the bus load.
4. **Monitor Crowd Density**: Check visual indicators to decide if you should board the bus.

## Future Enhancements
- **Authentication System**: Integrate a secure, multi-user login system.
- **Real-Time GPS Integration**: Incorporate real-time GPS for more accurate tracking.
- **Enhanced Predictive Models**: Experiment with different models for improved accuracy.
