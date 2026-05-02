"""
Road Hazard Detection System - Web Application (OOP Implementation)
A clean, maintainable web application using Object-Oriented Programming principles.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import HAZARD_CLASSES, TARGET_LOCATIONS
from src.feature_extraction import VisualFeatureExtractor
from src.model_trainer import RoadHazardClassifier

# ============================================================================
# Configuration & Enums
# ============================================================================

class HazardLevel(Enum):
    """Hazard classification levels."""
    SAFE = 0
    WET = 1
    SNOWY = 2
    ICY = 3
    STORM_RISK = 4


@dataclass
class WeatherData:
    """Weather conditions data structure."""
    temperature: float
    relative_humidity: float
    barometric_pressure: float
    dewpoint: float
    wind_speed: float
    visibility: float
    
    def to_array(self) -> np.ndarray:
        """Convert weather data to numpy array for model prediction."""
        return np.array([
            self.dewpoint,
            self.relative_humidity,
            self.barometric_pressure,
            self.temperature,
            self.wind_speed,
            self.visibility
        ]).reshape(1, -1)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WeatherData':
        """Create WeatherData instance from dictionary."""
        return cls(
            temperature=data.get('temperature', 20),
            relative_humidity=data.get('relativeHumidity', 60),
            barometric_pressure=data.get('barometricPressure', 1013),
            dewpoint=data.get('dewpoint', 10),
            wind_speed=data.get('windSpeed', 10),
            visibility=data.get('visibility', 10)
        )


@dataclass
class PredictionResult:
    """Prediction result data structure."""
    hazard_class: int
    hazard_label: str
    confidence: float
    confidence_scores: Dict[str, float]
    features_used: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'success': True,
            'hazard_class': self.hazard_class,
            'prediction': self.hazard_label,
            'confidence': round(self.confidence * 100, 2),
            'confidence_scores': self.confidence_scores,
            'features_used': self.features_used
        }


# ============================================================================
# Service Classes (Business Logic)
# ============================================================================

class ModelManager:
    """Manages machine learning models lifecycle."""
    
    def __init__(self):
        self._classifier: Optional[RoadHazardClassifier] = None
        self._visual_extractor: Optional[VisualFeatureExtractor] = None
        self._logger = logging.getLogger(__name__)
    
    @property
    def is_ready(self) -> bool:
        """Check if all models are loaded."""
        return self._classifier is not None and self._visual_extractor is not None
    
    def initialize(self) -> bool:
        """Load trained models at startup."""
        try:
            self._logger.info("Initializing models...")
            self._classifier = RoadHazardClassifier()
            self._visual_extractor = VisualFeatureExtractor()
            self._logger.info("✓ All models loaded successfully")
            return True
        except Exception as e:
            self._logger.error(f"✗ Error loading models: {e}")
            return False
    
    def get_classifier(self) -> Optional[RoadHazardClassifier]:
        """Get the hazard classifier model."""
        return self._classifier
    
    def get_visual_extractor(self) -> Optional[VisualFeatureExtractor]:
        """Get the visual feature extractor."""
        return self._visual_extractor


class PredictionService:
    """Handles prediction logic for road hazard detection."""
    
    def __init__(self, model_manager: ModelManager):
        self._model_manager = model_manager
        self._logger = logging.getLogger(__name__)
    
    def predict_from_image(self, image_file, weather_data: Optional[WeatherData] = None) -> Dict:
        """
        Make a prediction from an image with optional weather data.
        
        Args:
            image_file: File-like object containing the image
            weather_data: Optional weather conditions
        
        Returns:
            Dictionary with prediction results or error
        """
        try:
            visual_features = None
            weather_features = None
            features_used = []
            
            # Extract visual features from image
            extractor = self._model_manager.get_visual_extractor()
            if image_file and extractor:
                try:
                    visual_features = extractor.extract_from_file(image_file)
                    if visual_features is not None:
                        features_used.append('visual')
                        self._logger.info("Extracted visual features from image")
                except Exception as e:
                    self._logger.warning(f"Error extracting visual features: {e}")
            
            # Use provided weather data
            if weather_data:
                features_used.append('weather')
                weather_features = weather_data.to_array()
            
            # Make prediction
            if visual_features is not None or weather_features is not None:
                return self._make_prediction(
                    visual_features, 
                    weather_features, 
                    features_used
                )
            else:
                return {
                    'success': False,
                    'error': 'No valid features provided'
                }
        
        except Exception as e:
            self._logger.error(f"Unexpected error in prediction: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_from_weather(self, weather_data: WeatherData) -> Dict:
        """
        Make a prediction from weather data only.
        
        Args:
            weather_data: Weather conditions
        
        Returns:
            Dictionary with prediction results or error
        """
        try:
            weather_features = weather_data.to_array()
            return self._make_prediction(None, weather_features, ['weather'])
        except Exception as e:
            self._logger.error(f"Error in weather prediction: {e}")
            return {'success': False, 'error': str(e)}
    
    def _make_prediction(
        self, 
        visual_features: Optional[np.ndarray],
        weather_features: Optional[np.ndarray],
        features_used: List[str]
    ) -> Dict:
        """
        Internal method to make predictions using available features.
        
        Args:
            visual_features: Visual features from image (optional)
            weather_features: Weather features (optional)
            features_used: List of feature types used
        
        Returns:
            Dictionary with prediction results
        """
        classifier = self._model_manager.get_classifier()
        if not classifier:
            return {'success': False, 'error': 'Model not available'}
        
        try:
            # Combine features appropriately
            if visual_features is not None and weather_features is not None:
                features = np.hstack([visual_features.reshape(1, -1), weather_features])
            elif visual_features is not None:
                features = visual_features.reshape(1, -1)
            else:
                features = weather_features
            
            # Get prediction
            prediction = classifier.predict(features)
            confidence_scores = classifier.predict_proba(features)
            
            hazard_class = int(prediction[0])
            max_confidence = float(np.max(confidence_scores[0]))
            
            result = PredictionResult(
                hazard_class=hazard_class,
                hazard_label=HAZARD_CLASSES.get(hazard_class, 'unknown'),
                confidence=max_confidence,
                confidence_scores={
                    HAZARD_CLASSES[i]: round(float(confidence_scores[0][i]) * 100, 2)
                    for i in range(len(HAZARD_CLASSES))
                },
                features_used=features_used
            )
            
            self._logger.info(
                f"✓ Prediction: {result.hazard_label} ({result.confidence * 100:.1f}%)"
            )
            return result.to_dict()
        
        except Exception as e:
            self._logger.error(f"Error during prediction: {e}")
            return {'success': False, 'error': str(e)}


class LocationService:
    """Manages location data and operations."""
    
    def __init__(self):
        self._locations = self._initialize_locations()
        self._logger = logging.getLogger(__name__)
    
    def _initialize_locations(self) -> List[Dict]:
        """Initialize location data from config."""
        locations = []
        for name, coords in TARGET_LOCATIONS.items():
            locations.append({
                'id': name,
                'name': name.replace('_', ' ').title(),
                'latitude': coords['lat'],
                'longitude': coords['lon']
            })
        return locations
    
    def get_all_locations(self) -> List[Dict]:
        """Get all monitored locations."""
        return self._locations
    
    def get_location_by_id(self, location_id: str) -> Optional[Dict]:
        """Get specific location by ID."""
        return next((loc for loc in self._locations if loc['id'] == location_id), None)


class SystemHealth:
    """Manages system health and status information."""
    
    def __init__(self, model_manager: ModelManager):
        self._model_manager = model_manager
        self._version = "1.0.0"
    
    def get_status(self) -> Dict:
        """Get current system status."""
        return {
            'status': 'healthy',
            'model_loaded': self._model_manager.is_ready,
            'version': self._version
        }


# ============================================================================
# Flask Application Factory
# ============================================================================

class RoadSafetyApp:
    """Main application class - Flask app with integrated services."""
    
    def __init__(self):
        self.flask_app = Flask(__name__, template_folder='templates', static_folder='static')
        CORS(self.flask_app)
        
        # Initialize logging
        self._setup_logging()
        self._logger = logging.getLogger(__name__)
        
        # Initialize services
        self._model_manager = ModelManager()
        self._prediction_service = PredictionService(self._model_manager)
        self._location_service = LocationService()
        self._health_service = SystemHealth(self._model_manager)
        
        # Register routes
        self._register_routes()
    
    def _setup_logging(self):
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _register_routes(self):
        """Register all Flask routes."""
        # Main page
        self.flask_app.route('/')(self._route_index)
        
        # API routes
        self.flask_app.route('/api/locations', methods=['GET'])(self._route_get_locations)
        self.flask_app.route('/api/prediction', methods=['POST'])(self._route_predict)
        self.flask_app.route('/api/batch-prediction', methods=['POST'])(self._route_batch_predict)
        self.flask_app.route('/api/health', methods=['GET'])(self._route_health)
        
        # Error handlers
        self.flask_app.errorhandler(404)(self._handle_404)
        self.flask_app.errorhandler(500)(self._handle_500)
    
    # Route Handlers
    def _route_index(self):
        """Serve main dashboard."""
        return render_template('index.html')
    
    def _route_get_locations(self):
        """Get all monitored locations."""
        locations = self._location_service.get_all_locations()
        return jsonify(locations)
    
    def _route_predict(self):
        """Handle single prediction request."""
        try:
            image_file = request.files.get('image')
            weather_data_str = request.form.get('weather_data')
            
            weather_data = None
            if weather_data_str:
                try:
                    weather_dict = json.loads(weather_data_str)
                    weather_data = WeatherData.from_dict(weather_dict)
                except Exception as e:
                    self._logger.warning(f"Error parsing weather data: {e}")
            
            result = self._prediction_service.predict_from_image(image_file, weather_data)
            status_code = 200 if result.get('success') else 400
            return jsonify(result), status_code
        
        except Exception as e:
            self._logger.error(f"Error in prediction endpoint: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    def _route_batch_predict(self):
        """Handle batch prediction request."""
        try:
            data = request.get_json()
            predictions = []
            
            for item in data.get('items', []):
                location = item.get('location')
                weather_dict = item.get('weather')
                
                if weather_dict:
                    weather_data = WeatherData.from_dict(weather_dict)
                    result = self._prediction_service.predict_from_weather(weather_data)
                    
                    if result.get('success'):
                        predictions.append({
                            'location': location,
                            'hazard_class': result['hazard_class'],
                            'prediction': result['prediction'],
                            'confidence': result['confidence']
                        })
            
            return jsonify({'success': True, 'predictions': predictions})
        
        except Exception as e:
            self._logger.error(f"Error in batch prediction: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    def _route_health(self):
        """Health check endpoint."""
        return jsonify(self._health_service.get_status())
    
    def _handle_404(self, error):
        """Handle 404 errors."""
        return jsonify({'error': 'Not found'}), 404
    
    def _handle_500(self, error):
        """Handle 500 errors."""
        self._logger.error(f"Internal server error: {error}")
        return jsonify({'error': 'Internal server error'}), 500
    
    def initialize(self) -> bool:
        """Initialize the application."""
        self._logger.info("=" * 80)
        self._logger.info("ROAD SAFETY - WEB APPLICATION INITIALIZATION")
        self._logger.info("=" * 80)
        success = self._model_manager.initialize()
        if success:
            self._logger.info(f"✓ Application ready at http://localhost:5000")
        else:
            self._logger.warning("⚠ Application running but models not loaded")
        return success
    
    def run(self, debug: bool = True, host: str = '0.0.0.0', port: int = 5000):
        """Run the Flask application."""
        self.flask_app.run(debug=debug, host=host, port=port)


# ============================================================================
# Application Entry Point
# ============================================================================

def create_app() -> RoadSafetyApp:
    """Factory function to create application instance."""
    app = RoadSafetyApp()
    app.initialize()
    return app


# Create global app instance
road_safety_app = create_app()
app = road_safety_app.flask_app  # For WSGI servers


if __name__ == '__main__':
    road_safety_app.run(debug=True)
