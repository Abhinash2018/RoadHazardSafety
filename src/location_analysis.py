"""Live location analysis by combining local weather and camera metadata."""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

from config import HAZARD_CLASSES, TARGET_LOCATIONS

logger = logging.getLogger(__name__)


class LocationAnalysisService:
    """Fetches current local weather and prepares a location-level analysis."""

    def __init__(
        self, prediction_service, location_service, weather_data_factory, cache_seconds: int = 60
    ):
        self._prediction_service = prediction_service
        self._location_service = location_service
        self._weather_data_factory = weather_data_factory
        self._cache_seconds = cache_seconds
        self._cache: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "RoadSafety/1.0"})

    def analyze(self, location_id: str) -> Optional[Dict]:
        """Return a cached or freshly fetched analysis for a monitored location."""
        location = self._location_service.get_location_by_id(location_id)
        if not location:
            return None

        cached = self._cache.get(location_id)
        if cached and time.time() - cached["_cached_at"] < self._cache_seconds:
            return cached["data"]

        try:
            weather = self._fetch_weather(location)
            if self._prediction_service.supports_weather_only():
                prediction = self._prediction_service.predict_from_weather(
                    self._weather_data_factory(weather)
                )
            else:
                prediction = self._weather_risk_estimate(weather)
            result = self._build_result(location, weather, prediction)
            with self._lock:
                self._cache[location_id] = {"_cached_at": time.time(), "data": result}
            return result
        except requests.RequestException as error:
            logger.warning("Live weather request failed for %s: %s", location_id, error)
            return self._fallback_result(location, str(error))
        except Exception as error:
            logger.exception("Location analysis failed for %s", location_id)
            return self._fallback_result(location, str(error))

    def _fetch_weather(self, location: Dict):
        """Fetch current observations from Open-Meteo; this endpoint needs no key."""
        params = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": (
                "temperature_2m,relative_humidity_2m,dew_point_2m,pressure_msl,"
                "wind_speed_10m,visibility,precipitation,rain,showers,snowfall,weather_code"
            ),
            "wind_speed_unit": "kmh",
            "temperature_unit": "celsius",
            "timezone": "auto",
        }
        response = self._session.get(
            "https://api.open-meteo.com/v1/forecast", params=params, timeout=10
        )
        response.raise_for_status()
        current = response.json().get("current", {})
        return {
            "temperature": current.get("temperature_2m", 20),
            "relativeHumidity": current.get("relative_humidity_2m", 60),
            "barometricPressure": current.get("pressure_msl", 1013),
            "dewpoint": current.get("dew_point_2m", 10),
            "windSpeed": current.get("wind_speed_10m", 10),
            "visibility": (current.get("visibility", 10000) or 10000) / 1000,
            "precipitation": current.get("precipitation", 0) or 0,
            "rain": current.get("rain", 0) or 0,
            "showers": current.get("showers", 0) or 0,
            "snowfall": current.get("snowfall", 0) or 0,
            "weatherCode": current.get("weather_code"),
            "observedAt": current.get("time"),
            "source": "Open-Meteo",
        }

    def _build_result(self, location: Dict, weather: Dict, prediction: Dict) -> Dict:
        hazard = prediction.get("prediction", "unknown")
        return {
            "success": True,
            "location": location,
            "weather": weather,
            "prediction": prediction,
            "camera": self._camera_metadata(location["id"]),
            "analysis": self._explain(hazard, weather),
            "generatedAt": datetime.now(timezone.utc).isoformat(),
        }

    def _camera_metadata(self, location_id: str) -> Dict:
        """Expose a configured feed without inventing a camera URL."""
        env_name = f"CAMERA_FEED_{location_id.upper()}"
        image_url = os.getenv(env_name)
        return {
            "available": bool(image_url),
            "imageUrl": image_url,
            "source": "Configured camera feed" if image_url else "Not configured",
        }

    @staticmethod
    def _explain(hazard: str, weather: Dict) -> str:
        reasons = []
        if weather["visibility"] < 5:
            reasons.append("reduced visibility")
        if weather["relativeHumidity"] >= 85:
            reasons.append("high humidity")
        if weather["rain"] or weather["showers"] or weather["snowfall"]:
            reasons.append("active precipitation")
        if weather["windSpeed"] >= 35:
            reasons.append("strong wind")
        if weather["temperature"] <= 1 and weather["relativeHumidity"] >= 80:
            reasons.append("possible icing conditions")
        if not reasons:
            reasons.append("no major atmospheric warning detected")
        return f"Current estimate is {hazard.replace('_', ' ')} because of {', '.join(reasons)}."

    @staticmethod
    def _weather_risk_estimate(weather: Dict) -> Dict:
        """Provide a transparent weather-only estimate when no compatible model exists."""
        if weather["snowfall"] > 0 or (
            weather["temperature"] <= 0 and weather["relativeHumidity"] >= 80
        ):
            hazard = "icy" if weather["temperature"] <= 0 else "snowy"
        elif weather["windSpeed"] >= 40 or weather["visibility"] < 2:
            hazard = "storm_risk"
        elif weather["rain"] > 0 or weather["showers"] > 0 or weather["relativeHumidity"] >= 85:
            hazard = "wet"
        else:
            hazard = "safe"

        confidence = {
            "safe": 0.72,
            "wet": 0.78,
            "snowy": 0.82,
            "icy": 0.84,
            "storm_risk": 0.80,
        }[hazard]
        scores = {label: 0.0 for label in HAZARD_CLASSES.values()}
        scores[hazard] = round(confidence * 100, 2)
        return {
            "success": True,
            "hazard_class": list(HAZARD_CLASSES.keys())[list(HAZARD_CLASSES.values()).index(hazard)],
            "prediction": hazard,
            "confidence": round(confidence * 100, 2),
            "confidence_scores": scores,
            "features_used": ["weather"],
            "method": "weather-rule-fallback",
        }

    @staticmethod
    def _fallback_result(location: Dict, error: str) -> Dict:
        return {
            "success": False,
            "location": location,
            "error": "Live weather is temporarily unavailable.",
            "detail": error,
            "camera": {"available": False, "imageUrl": None, "source": "Not configured"},
        }