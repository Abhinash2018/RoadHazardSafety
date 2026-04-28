"""
Data acquisition module for fetching TxDOT camera images and NOAA weather data.
Handles synchronization between camera images and atmospheric telemetry.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Tuple, Dict, List
import logging

from config import (
    TXDOT_CAMERA_API,
    NOAA_API_BASE,
    CAMERA_FETCH_INTERVAL_MINUTES,
    WEATHER_FETCH_INTERVAL_MINUTES,
    TARGET_LOCATIONS,
    WEATHER_FEATURES,
    DATA_DIR,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TxDOTCameraFetcher:
    """Fetches traffic camera images from TxDOT."""

    def __init__(self, api_endpoint: str = TXDOT_CAMERA_API):
        self.api_endpoint = api_endpoint
        self.session = requests.Session()

    def fetch_camera_images(self, location: str, num_images: int = 10) -> List[Dict]:
        """
        Fetch camera images from TxDOT API.

        Args:
            location: Location identifier (e.g., 'austin_north')
            num_images: Number of recent images to fetch

        Returns:
            List of image metadata with timestamps and URLs
        """
        try:
            # This is a placeholder implementation
            # In production, use actual TxDOT API authentication and endpoints
            logger.info(f"Fetching {num_images} camera images from {location}")

            # Placeholder response structure
            images = []
            for i in range(num_images):
                timestamp = datetime.now() - timedelta(minutes=i * CAMERA_FETCH_INTERVAL_MINUTES)
                images.append(
                    {
                        "location": location,
                        "timestamp": timestamp,
                        "image_url": f"https://placeholder.txdot.gov/camera_{location}_{i}.jpg",
                        "camera_id": f"cam_{location}_{i}",
                    }
                )

            logger.info(f"Successfully fetched {len(images)} images")
            return images

        except Exception as e:
            logger.error(f"Error fetching TxDOT images: {e}")
            return []

    def download_image(self, image_url: str, save_path: str) -> bool:
        """Download image from URL and save locally."""
        try:
            response = self.session.get(image_url, timeout=10)
            if response.status_code == 200:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Downloaded image to {save_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error downloading image from {image_url}: {e}")
            return False


class NOAAWeatherFetcher:
    """Fetches atmospheric telemetry from NOAA."""

    def __init__(self, api_base: str = NOAA_API_BASE):
        self.api_base = api_base
        self.session = requests.Session()

    def get_nearest_weather_station(self, lat: float, lon: float) -> Dict:
        """
        Find nearest NOAA weather station to given coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Weather station metadata
        """
        try:
            # NOAA API to get grid point and station data
            points_url = f"{self.api_base}/points/{lat},{lon}"
            response = self.session.get(points_url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                # Extract nearest weather station URL from the response
                station_url = data.get("properties", {}).get("observationStations")
                logger.info(f"Found weather station at {points_url}")
                return {"url": station_url, "lat": lat, "lon": lon}

            return None
        except Exception as e:
            logger.error(f"Error finding weather station for ({lat}, {lon}): {e}")
            return None

    def fetch_weather_data(
        self, lat: float, lon: float, hours_back: int = 1
    ) -> pd.DataFrame:
        """
        Fetch recent weather observations from NOAA.

        Args:
            lat: Latitude
            lon: Longitude
            hours_back: Number of hours of historical data to fetch

        Returns:
            DataFrame with weather observations
        """
        try:
            # Get nearest station
            station = self.get_nearest_weather_station(lat, lon)
            if not station:
                logger.warning(f"No station found for ({lat}, {lon})")
                return pd.DataFrame()

            # Placeholder implementation - fetch observations
            logger.info(f"Fetching weather data for station at ({lat}, {lon})")

            # Create placeholder weather data
            timestamps = pd.date_range(
                end=datetime.now(), periods=hours_back * 12, freq="5min"
            )
            weather_data = {
                "timestamp": timestamps,
                "latitude": [lat] * len(timestamps),
                "longitude": [lon] * len(timestamps),
                "dewpoint": np.random.uniform(5, 25, len(timestamps)),
                "relativeHumidity": np.random.uniform(30, 90, len(timestamps)),
                "barometricPressure": np.random.uniform(1010, 1030, len(timestamps)),
                "temperature": np.random.uniform(10, 30, len(timestamps)),
                "windSpeed": np.random.uniform(0, 20, len(timestamps)),
                "visibility": np.random.uniform(5, 20, len(timestamps)),
            }

            df = pd.DataFrame(weather_data)
            logger.info(f"Fetched {len(df)} weather observations")
            return df

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return pd.DataFrame()


class DataSynchronizer:
    """Synchronizes camera images with weather data by timestamp."""

    def __init__(self):
        self.camera_fetcher = TxDOTCameraFetcher()
        self.weather_fetcher = NOAAWeatherFetcher()

    def synchronize(
        self, location: str, lat: float, lon: float, time_window_minutes: int = 5
    ) -> pd.DataFrame:
        """
        Fetch and synchronize camera images with weather data.

        Args:
            location: Location identifier
            lat: Latitude
            lon: Longitude
            time_window_minutes: Tolerance window for matching timestamps

        Returns:
            DataFrame with synchronized camera and weather data
        """
        logger.info(f"Synchronizing data for {location}...")

        # Fetch camera images
        images = self.camera_fetcher.fetch_camera_images(location, num_images=10)

        # Fetch weather data
        weather_df = self.weather_fetcher.fetch_weather_data(lat, lon, hours_back=1)

        if not images or weather_df.empty:
            logger.warning("Could not fetch complete data")
            return pd.DataFrame()

        # Convert images to DataFrame for easier merging
        image_times = [img["timestamp"] for img in images]
        image_df = pd.DataFrame(
            {
                "timestamp": image_times,
                "image_url": [img["image_url"] for img in images],
                "camera_id": [img["camera_id"] for img in images],
            }
        )

        # Merge on closest timestamp (within time_window_minutes)
        image_df.set_index("timestamp", inplace=True)
        weather_df.set_index("timestamp", inplace=True)

        # Use nearest match within time window
        merged = pd.merge_asof(
            image_df.reset_index().sort_values("timestamp"),
            weather_df.reset_index().sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=time_window_minutes),
        )

        logger.info(f"Synchronized {len(merged)} records")
        return merged

    def fetch_all_locations(self) -> pd.DataFrame:
        """Fetch and synchronize data for all target locations."""
        all_data = []

        for location, coords in TARGET_LOCATIONS.items():
            data = self.synchronize(location, coords["lat"], coords["lon"])
            if not data.empty:
                all_data.append(data)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined {len(combined)} records from all locations")
            return combined

        return pd.DataFrame()

    def save_synchronized_data(self, df: pd.DataFrame, filename: str = "synchronized_data.csv"):
        """Save synchronized data to CSV."""
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved synchronized data to {filepath}")
        return filepath


if __name__ == "__main__":
    # Example usage
    synchronizer = DataSynchronizer()
    data = synchronizer.fetch_all_locations()
    if not data.empty:
        synchronizer.save_synchronized_data(data)
        print(data.head())
    else:
        print("No data fetched")
