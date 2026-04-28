"""
Configuration and constants for the Road Hazard Detection System
"""

# API Endpoints
TXDOT_CAMERA_API = "https://www.txdot.gov/api/cameras"  # Placeholder - replace with actual endpoint
NOAA_API_BASE = "https://api.weather.gov"

# Data collection parameters
CAMERA_FETCH_INTERVAL_MINUTES = 5  # Fetch TxDOT camera images every 5-10 minutes
WEATHER_FETCH_INTERVAL_MINUTES = 5

# Geographic focus: I-35 corridor (Austin to San Marcos)
TARGET_LOCATIONS = {
    "austin_north": {"lat": 30.3667, "lon": -97.7333},
    "austin_south": {"lat": 30.2671, "lon": -97.7430},
    "san_marcos": {"lat": 29.8833, "lon": -97.9414},
}

# Weather features to extract from NOAA
WEATHER_FEATURES = [
    "dewpoint",
    "relativeHumidity",
    "barometricPressure",
    "temperature",
    "windSpeed",
    "visibility",
]

# Road hazard classification labels
HAZARD_CLASSES = {
    0: "safe",
    1: "wet",
    2: "snowy",
    3: "icy",
    4: "storm_risk",
}

# Model parameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}

# Cross-validation parameters
CV_FOLDS = 5
TEST_SPLIT = 0.2

# Output paths
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
