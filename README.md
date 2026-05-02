# 🛣️ Road Hazard Detection System

A comprehensive machine learning system for detecting and predicting road hazards using real-time camera imagery and weather data along the I-35 corridor (Austin to San Marcos, Texas).

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Flask](https://img.shields.io/badge/Flask-Web%20App-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Road Hazard Detection System is an intelligent platform that combines computer vision and weather data to identify hazardous road conditions in real-time. The system continuously monitors road conditions across multiple locations and provides predictions for five hazard categories: **safe**, **wet**, **snowy**, **icy**, and **storm risk**.

### Key Capabilities

- 🎥 **Visual Analysis**: Processes camera images to extract visual features indicative of road conditions
- 🌦️ **Weather Integration**: Incorporates real-time weather data (humidity, temperature, pressure, visibility, wind speed)
- 🤖 **Dual Model Architecture**: Operates both camera-only and fused (visual + weather) models for flexible predictions
- 📊 **Interactive Dashboard**: Web-based interface for hazard analysis and historical tracking
- 🎯 **Confidence Scoring**: Provides probability estimates for each hazard classification

## Features

### Core ML Features
- **Multi-modal Learning**: Combines visual features from road images with meteorological data
- **Camera-Only Analysis**: Standalone predictions from road imagery
- **Fused Model Predictions**: Enhanced accuracy by combining visual and weather features
- **Confidence Scoring**: Probabilistic predictions for decision-making

### Web Dashboard Features
- **Real-time Hazard Detection**: Upload images or input weather conditions for instant analysis
- **Multi-location Monitoring**: Track 3 key locations along the I-35 corridor:
  - Austin North
  - Austin South
  - San Marcos
- **Interactive Maps**: Visualize monitored locations and their current conditions
- **Prediction History**: Review recent predictions with confidence scores and timestamps
- **System Status**: Monitor active models and system health
- **Weather Conditions Input**: Manually enter weather parameters for prediction

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Sources                            │
│  ┌──────────────────┐         ┌──────────────────────┐      │
│  │  TxDOT Cameras   │         │  NOAA Weather API    │      │
│  └──────────────────┘         └──────────────────────┘      │
└────────────────┬────────────────────────────────┬────────────┘
                 │                                │
                 ▼                                ▼
        ┌─────────────────┐         ┌──────────────────────┐
        │  Visual Feature │         │ Weather Feature      │
        │  Extractor      │         │ Extractor            │
        │  (CNN/Colors)   │         │ (Normalization)      │
        └────────┬────────┘         └──────────┬───────────┘
                 │                             │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │   Feature Fusion Layer      │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────────────┐
                 │  Trained Models                      │
                 │  ┌────────────────────────────────┐ │
                 │  │  RandomForest (Fused)          │ │
                 │  │  RandomForest (Camera-Only)    │ │
                 │  └────────────────────────────────┘ │
                 └──────────────┬──────────────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │   Output (5 Classes)        │
                 │  ┌────────────────────┐    │
                 │  │ • Safe             │    │
                 │  │ • Wet              │    │
                 │  │ • Snowy            │    │
                 │  │ • Icy              │    │
                 │  │ • Storm Risk       │    │
                 │  └────────────────────┘    │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │  Flask Web Application      │
                 │  • Dashboard                │
                 │  • Analytics                │
                 │  • API Endpoints            │
                 └─────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- 2GB+ disk space for models and datasets
- Windows, macOS, or Linux

### Setup Instructions

#### Option 1: Automated Setup (Windows)

```bash
# Double-click setup_website.bat in the project root
```

#### Option 2: Manual Setup (All Platforms)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RoadSafety
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv RoadSafety
   ```

3. **Activate the virtual environment**
   
   **Windows:**
   ```bash
   RoadSafety\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source RoadSafety/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Running the Web Dashboard

#### Windows (Automated)
```bash
# Double-click run_website.bat
# The dashboard automatically opens at http://localhost:5000
```

#### Command Line (All Platforms)
```bash
python app.py
```

Then open your browser and navigate to: **http://localhost:5000**

### Using the Dashboard

1. **Upload an Image**
   - Select a road image from your computer
   - Click "Analyze" to get camera-only predictions

2. **Input Weather Data**
   - Select a location
   - Enter weather conditions (temperature, humidity, etc.)
   - Get fused model predictions

3. **View Predictions**
   - See confidence scores for each hazard class
   - Review prediction history
   - Monitor multiple locations

## Usage

### Web Interface

Start the Flask server:
```bash
python app.py
```

The dashboard provides an intuitive interface for:
- Uploading road images for analysis
- Entering weather conditions
- Viewing real-time predictions with confidence scores
- Monitoring multiple locations along I-35
- Tracking prediction history

### Python API (CLI)

Run the main orchestrator:

```bash
python main.py
```

This will:
1. Create/load a synthetic dataset
2. Extract visual features from images
3. Extract weather features
4. Train both camera-only and fused models
5. Evaluate model performance
6. Generate detailed classification reports

### Module Usage

```python
from src.model_trainer import RoadHazardClassifier
from src.feature_extraction import VisualFeatureExtractor

# Initialize extractor and model
extractor = VisualFeatureExtractor()
classifier = RoadHazardClassifier()

# Extract features from an image
visual_features = extractor.extract_from_image('road_image.jpg')

# Make predictions
hazard_class = classifier.predict(visual_features)
confidence_scores = classifier.predict_proba(visual_features)

print(f"Predicted hazard: {hazard_class}")
print(f"Confidence: {max(confidence_scores):.2%}")
```

## Project Structure

```
RoadSafety/
├── app.py                          # Flask web application entry point
├── main.py                         # ML pipeline orchestrator
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── QUICK_START.md                  # Quick start guide
├── WEBSITE_README.md               # Web dashboard documentation
├── run_website.bat                 # Windows script to run web app
├── setup_website.bat               # Windows setup script
├── setup_website.sh                # Unix setup script
│
├── src/                            # Core ML modules
│   ├── __init__.py
│   ├── config.py                   # Configuration and constants
│   ├── data_acquisition.py         # API integration (TxDOT, NOAA)
│   ├── feature_extraction.py       # Visual & weather feature extraction
│   ├── model_trainer.py            # Model training and comparison
│   └── evaluator.py                # Model evaluation and reporting
│
├── static/                         # Web assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── templates/                      # HTML templates
│   └── index.html
│
├── data/                           # Data directory
│   └── synchronized_data.csv       # Training dataset
│
├── models/                         # Trained models
│   ├── camera_only_model.pkl
│   └── fused_model.pkl
│
└── results/                        # Evaluation results
    ├── classification_report_camera_only_*.txt
    ├── classification_report_fused_model_*.txt
    └── summary_report_*.txt
```

## Model Architecture

### Visual Feature Extractor
- Processes road images using computer vision techniques
- Extracts color histograms, edge detection, and texture features
- Outputs standardized feature vectors for ML models

### Weather Feature Extractor
- Normalizes meteorological data from NOAA API
- Features: temperature, humidity, pressure, visibility, wind speed, dew point
- Handles missing data and outliers

### Classification Models

#### Camera-Only Model
- **Type**: Random Forest Classifier
- **Input**: Visual features (128-dimensional vectors)
- **Output**: 5-class hazard prediction with confidence scores
- **Configuration**: 100 trees, max depth 20, optimized for interpretability

#### Fused Model
- **Type**: Random Forest Classifier
- **Input**: Combined visual features + weather features
- **Output**: 5-class hazard prediction with confidence scores
- **Advantage**: Better accuracy by leveraging multiple data sources

### Training Pipeline

```
Raw Data → Feature Extraction → Feature Fusion → Model Training → Cross-Validation → Evaluation
```

The system uses 5-fold cross-validation and an 80/20 train-test split for robust performance estimation.

## Performance

The models are evaluated on:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean for balanced evaluation
- **Confusion Matrix**: Detailed error analysis

Results are saved in the `results/` directory with timestamps.

## Configuration

Key parameters can be modified in [src/config.py](src/config.py):

```python
# API update intervals (minutes)
CAMERA_FETCH_INTERVAL_MINUTES = 5
WEATHER_FETCH_INTERVAL_MINUTES = 5

# Monitored locations
TARGET_LOCATIONS = {
    "austin_north": {"lat": 30.3667, "lon": -97.7333},
    "austin_south": {"lat": 30.2671, "lon": -97.7430},
    "san_marcos": {"lat": 29.8833, "lon": -97.9414},
}

# Model hyperparameters
RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}
```

## API Endpoints (Web Dashboard)

The Flask application provides the following endpoints:

- `GET /` - Main dashboard page
- `POST /api/predict-image` - Analyze uploaded road image
- `POST /api/predict-weather` - Get predictions based on weather data
- `GET /api/history` - Retrieve prediction history
- `GET /api/locations` - Get monitored locations
- `GET /api/status` - System health and model status

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## Future Enhancements

- [ ] Real-time API integration with TxDOT and NOAA
- [ ] Deep learning models (CNN, RNN) for improved accuracy
- [ ] Mobile app for on-road notifications
- [ ] Database integration for large-scale data storage
- [ ] Automated model retraining pipeline
- [ ] Advanced analytics and trend analysis
- [ ] Alert system for hazardous conditions
- [ ] Multi-language support

## Troubleshooting

### Port 5000 Already in Use
```bash
# Use a different port
python app.py --port 5001
```

### Models Not Found
Ensure trained models exist in the `models/` directory. Run `main.py` to train new models if needed.

### Dependency Issues
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact & Support

For questions or issues, please:
- Open an issue on GitHub
- Contact the development team
- Check the [documentation](WEBSITE_README.md)

## Acknowledgments

- TxDOT for camera data infrastructure
- NOAA for weather data services
- scikit-learn and OpenCV communities
- Contributors and maintainers

---

**Last Updated**: May 2026  
**Version**: 1.0.0  
**Status**: Active Development
