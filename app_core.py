import joblib
import pandas as pd
import numpy as np

def load_model(model_path):
    return joblib.load(model_path)

def ensure_features_in_data(data, expected_features):
    # Add any missing features with NaN values and reorder columns to match expected features
    for feature in expected_features:
        if feature not in data.columns:
            data[feature] = np.nan
    return data[expected_features]

def fill_missing_values_with_zero(data):
    # Fill any missing values with 0
    return data.fillna(0)

def predict_delay(model, data):
    return model.predict(data)

def fill_rainfall_based_on_forecast(df, historical_medians):
    rainy_forecasts = [
        "Light Rain", "Moderate Rain", "Heavy Rain", "Passing Showers",
        "Light Showers", "Showers", "Heavy Showers", "Thundery Showers",
        "Heavy Thundery Showers", "Heavy Thundery Showers with Gusty Winds"
    ]

    rainy_mask = df['forecast_text'].isin(rainy_forecasts)
    non_rainy_mask = ~rainy_mask
    
    df.loc[non_rainy_mask, ['Daily Rainfall Total (mm)', 'Highest 30 Min Rainfall (mm)',
                            'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)']] = 0
    
    df.loc[rainy_mask, ['Daily Rainfall Total (mm)', 'Highest 30 Min Rainfall (mm)',
                        'Highest 60 Min Rainfall (mm)', 'Highest 120 Min Rainfall (mm)']] = historical_medians
    
    return df
