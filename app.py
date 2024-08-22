import streamlit as st
import pandas as pd
import joblib
import requests
from datetime import datetime, timedelta

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Changi Flight Delay Predictor")

# Custom CSS for enhanced UI/UX focused on Singapore Changi Airport
st.markdown("""
    <style>
    body {
        background-color: #F8F8F8;  /* Ivory */
        color: #696969;  /* Dim Gray for text */
        font-family: 'Roboto', sans-serif;
        overflow: hidden; /* Prevent scrolling */
    }
    .main-header {
        background-color: #2E3B55;  /* Dark Slate Blue */
        padding: 15px;
        border-radius: 10px;
        color: #FFFFFF;  /* White */
        text-align: center;
        margin-bottom: 15px;
        font-weight: bold;  /* Ensures header text stands out */
        font-size: 22px;  /* Adjust font size for better visibility */
    }
    .sub-header {
        background-color: #708090;  /* Light Slate Gray */
        padding: 10px;
        border-radius: 10px;
        color: #FFFFFF;  /* White */
        text-align: center;
        margin-bottom: 15px;
        font-size: 16px;  /* Slightly smaller for sub-header */
    }
    .prediction-card {
        background-color: #FFFFFF;  /* White */
        border-radius: 10px;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
        padding: 15px;
        margin-top: 15px;
    }
    .weather-info {
        font-size: 20px;
        font-weight: bold;
        color: #2E3B55;  /* Dark Slate Blue */
        margin-bottom: 10px;
    }
    .weather-summary {
        font-size: 16px;
        color: #696969;  /* Dim Gray */
        margin-bottom: 20px;
    }
    .success-box {
        background-color: #D3D3D3;  /* Light Gray */
        padding: 10px;
        border-radius: 10px;
        color: #2E3B55;  /* Dark Slate Blue */
        font-weight: bold;
        font-size: 16px;
        text-align: center;
    }
    .error-box {
        background-color: #B0C4DE;  /* Light Steel Blue */
        padding: 10px;
        border-radius: 10px;
        color: #2E3B55;  /* Dark Slate Blue */
        font-weight: bold;
        font-size: 16px;
        text-align: center;
    }
    .stButton button {
        background-color: #4682B4;  /* Steel Blue */
        color: #FFFFFF;  /* White */
        font-size: 16px;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 15px;
        margin-top: 15px;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #708090;  /* Light Slate Gray */
    }
    .stTextInput, .stSelectbox {
        border-radius: 5px;
        padding: 8px;
        font-size: 14px;
        border: 1px solid #B0C4DE;  /* Light Steel Blue border */
        background-color: #FFFFFF;  /* White */
    }
    .loading-spinner {
        display: block;
        margin: 40px auto;
        width: 60px;
        height: 60px;
        border: 6px solid #f3f3f3;  /* Light spinner */
        border-top: 6px solid #4682B4;  /* Steel Blue spinner */
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    footer {
        background-color: #2E3B55;  /* Dark Slate Blue */
        color: #FFFFFF;  /* White */
        padding: 10px;
        text-align: center;
        border-radius: 0px 0px 10px 10px;
        margin-top: 15px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

# Load Models and Data
@st.cache_data
def load_iata_airport_list():
    iata_airport_path = './IATA_List_of_level_2_3_Airports.csv'
    iata_airport_list = pd.read_csv(iata_airport_path)
    return iata_airport_list

@st.cache_resource
def load_models():
    enhanced_model = joblib.load('./enhanced_model.pkl')
    return enhanced_model

# Weather API Integration
url_24hr = 'https://api-open.data.gov.sg/v2/real-time/api/twenty-four-hr-forecast'
url_4day = 'https://api-open.data.gov.sg/v2/real-time/api/four-day-outlook'

def fetch_weather_forecast_24hr():
    response = requests.get(url_24hr)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and 'records' in data['data']:
            return data['data']['records']
        else:
            st.warning("24-hour forecast data not found. Please try again later.")
            return []
    else:
        st.error(f"Failed to fetch 24-hour forecast data: {response.status_code}. Please check your connection and try again.")
        return []

def fetch_weather_forecast_4day():
    response = requests.get(url_4day)
    if response.status_code == 200:
        data = response.json()
        if 'data' in data and 'records' in data['data']:
            return data['data']['records']
        else:
            st.warning("4-day forecast data not found. Please try again later.")
            return []
    else:
        st.error(f"Failed to fetch 4-day forecast data: {response.status_code}. Please check your connection and try again.")
        return []

def get_weather_summary(weather_data_24hr, weather_data_4day, selected_datetime):
    for record in weather_data_24hr:
        for period in record.get('periods', []):
            start = period['timePeriod']['start']
            end = period['timePeriod']['end']
            start_time = datetime.fromisoformat(start[:-6])
            end_time = datetime.fromisoformat(end[:-6])
            if start_time <= selected_datetime <= end_time:
                return period['regions']['east']['text']

    for record in weather_data_4day:
        for forecast in record.get('forecasts', []):
            forecast_time = datetime.fromisoformat(forecast['timestamp'][:-6])
            if forecast_time.date() == selected_datetime.date():
                return forecast['forecast']['summary']

    return "No weather data available."

def determine_date_range(weather_data_24hr, weather_data_4day):
    earliest_date = datetime.today().date()
    latest_date = datetime.today().date()
    
    if weather_data_24hr:
        for record in weather_data_24hr:
            for period in record.get('periods', []):
                start = period['timePeriod']['start']
                end = period['timePeriod']['end']
                start_time = datetime.fromisoformat(start[:-6]).date()
                end_time = datetime.fromisoformat(end[:-6]).date()
                earliest_date = min(earliest_date, start_time)
                latest_date = max(latest_date, end_time)
    
    if weather_data_4day:
        for record in weather_data_4day:
            for forecast in record.get('forecasts', []):
                forecast_time = datetime.fromisoformat(forecast['timestamp'][:-6]).date()
                earliest_date = min(earliest_date, forecast_time)
                latest_date = max(latest_date, forecast_time)
    
    return earliest_date, latest_date

# Streamlit App
def main():
    # Display main header
    st.markdown("""
        <div class='main-header' style='color: #FFFFFF;'>
            <h1 style='color: #FFFFFF;'>✈️ Singapore Changi Airport Flight Delay Prediction</h1>
        </div>
    """, unsafe_allow_html=True)

    # Display sub-header with additional information
    st.markdown("""
        <div class='sub-header'>
            <p>Accurate predictions for flights departing from Singapore Changi Airport. Stay updated with real-time weather.</p>
        </div>
    """, unsafe_allow_html=True)

    # Fetch and process weather data to determine valid date range
    weather_data_24hr = fetch_weather_forecast_24hr()
    weather_data_4day = fetch_weather_forecast_4day()
    earliest_date, latest_date = determine_date_range(weather_data_24hr, weather_data_4day)
    
    st.header("Enter Your Flight Details")
    
    selected_date = st.date_input(
        "Departure Date",
        value=datetime.today().date(),
        min_value=earliest_date,
        max_value=latest_date
    )
    
    selected_time = st.time_input("Departure Time")
    
    if selected_date == datetime.today().date() and selected_time < datetime.now().time():
        st.warning("Selected time is in the past. Please choose a valid departure time.")
    
    iata_airport_list = load_iata_airport_list()
    airport_options = [
        f"{row['Airport Code']} ({row['Country']})"
        for _, row in iata_airport_list.iterrows()
    ]
    
    destination_airport = st.selectbox(
        "Destination Airport",
        options=airport_options,
        help="Choose your destination airport."
    )
    
    selected_airport_code = destination_airport.split(" ")[0]
    
    if weather_data_24hr or weather_data_4day:
        selected_datetime = datetime.combine(selected_date, selected_time)
        weather_summary = get_weather_summary(weather_data_24hr, weather_data_4day, selected_datetime)
        
        st.markdown(f"<div class='weather-info'>🌤️ Weather Information</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='weather-summary'>Weather on {selected_datetime.date()}: {weather_summary}</div>", unsafe_allow_html=True)
    else:
        st.warning("No weather data available.")
    
    if st.button('Predict Flight Delay'):
        if selected_date and selected_time and destination_airport:
            if selected_date == datetime.today().date() and selected_time < datetime.now().time():
                st.error("Selected time is in the past. Please choose a valid departure time.")
                return

            selected_datetime = datetime.combine(selected_date, selected_time)
            
            selected_airport_info = iata_airport_list.loc[
                iata_airport_list['Airport Code'] == selected_airport_code
            ]
            
            if not selected_airport_info.empty:
                scheduled_block_minutes = selected_airport_info['scheduledBlockMinutes'].values[0]
                air_minutes = selected_airport_info['airMinutes'].values[0]

                input_features = {
                    'DayOfWeek': selected_datetime.weekday(),
                    'scheduledBlockMinutes': scheduled_block_minutes,
                    'airMinutes': air_minutes,
                    'arrivalGateDelayMinutes': 15,
                    'departureRunwayDelayMinutes': 5,
                }
                
                input_data = pd.DataFrame([input_features])
                
                try:
                    enhanced_model = load_models()
                    prediction = enhanced_model.predict(input_data)
                    delay_minutes = prediction[0]
                    
                    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                    if delay_minutes < 0.5:
                        st.markdown("<div class='success-box'>Your flight is likely to be on time or experience minimal delay.</div>", unsafe_allow_html=True)
                    else:
                        delay_minutes = round(delay_minutes)
                        st.markdown(f"<div class='error-box'>This flight is predicted to be delayed by approximately {delay_minutes} minutes. Consider rescheduling or planning for the delay.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}. Please try again later.")
            else:
                st.error("Unable to find airport information. Please check the destination airport code.")
        else:
            st.error("All inputs are required: Departure Date, Time, and Destination Airport.")

    # Footer with additional information
    st.markdown("""
        <footer>
            <p>Singapore Changi Airport - Ranked among the world's best airports, handling over 60 million passengers annually.</p>
        </footer>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
