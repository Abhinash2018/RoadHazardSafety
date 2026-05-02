# Road Safety - Web Dashboard

A modern web application for the Road Hazard Detection System that enables users to analyze road conditions using machine learning.

## Features

- **Real-time Hazard Detection**: Upload road images or provide weather data to get instant hazard predictions
- **Multi-location Monitoring**: Track road conditions across the I-35 corridor (Austin North, Austin South, San Marcos)
- **Fused Model Predictions**: Combine visual features (from images) with weather data for better accuracy
- **Interactive Dashboard**: Beautiful, responsive interface with live system status monitoring
- **Prediction History**: View recent predictions and confidence scores
- **Historical Analytics**: Track statistics and trends over time

## Installation

### 1. Install Dependencies

The required packages are already listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

This will install:
- Flask & Flask-CORS (web framework)
- All existing ML dependencies (scikit-learn, pandas, etc.)

### 2. Verify Your Models

Make sure you have trained models in the `models/` directory. The app will load them on startup.

## Running the Application

### Start the Flask Web Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Open in Browser

Navigate to: **http://localhost:5000**

You'll see the Road Safety Dashboard with:
- Location monitoring cards
- Image upload tool
- Weather analysis tool
- Prediction history

## How to Use

### 1. Upload Road Image for Analysis

1. Click **"Upload Image"** in the Analysis Tools section
2. Drag and drop an image or click to browse
3. (Optional) Check the "Include Weather Data" checkbox and enter current weather conditions
4. Click **"Analyze"**
5. Get instant hazard prediction with confidence score

### 2. Weather-Based Hazard Analysis

1. Click **"Weather Analysis"** in the Analysis Tools section
2. Select a monitored location
3. Enter weather conditions:
   - Temperature (°C)
   - Humidity (%)
   - Wind Speed (km/h)
   - Visibility (km)
   - Dewpoint (°C)
   - Barometric Pressure (hPa)
4. Click **"Predict Hazard Level"**
5. View the predicted road condition and confidence

### 3. View Statistics

1. Click **"Historical Data"** to see:
   - Total predictions made
   - Safe condition count
   - Critical alerts
   - Average confidence

## Road Hazard Classes

The system classifies road conditions into 5 categories:

| Class | Color | Meaning |
|-------|-------|---------|
| **safe** | Green | Normal driving conditions |
| **wet** | Blue | Wet roads, reduced grip |
| **snowy** | Purple | Snow coverage on road |
| **icy** | Cyan | Icy conditions, very slippery |
| **storm_risk** | Red | Severe weather, high risk |

## API Endpoints

### GET `/api/locations`
Get all monitored locations.

**Response:**
```json
[
  {
    "id": "austin_north",
    "name": "Austin North",
    "latitude": 30.3667,
    "longitude": -97.7333
  }
]
```

### POST `/api/prediction`
Make a prediction from image and/or weather data.

**Parameters:**
- `image` (file): Road image file
- `weather_data` (JSON): Weather conditions

**Response:**
```json
{
  "success": true,
  "prediction": "wet",
  "hazard_class": 1,
  "confidence": 89.5,
  "features_used": ["visual", "weather"],
  "confidence_scores": {
    "safe": 5.2,
    "wet": 89.5,
    "snowy": 2.1,
    "icy": 1.8,
    "storm_risk": 1.4
  }
}
```

### POST `/api/batch-prediction`
Make predictions for multiple locations.

**Request:**
```json
{
  "items": [
    {
      "location": "austin_north",
      "weather": {
        "temperature": 15,
        "relativeHumidity": 75,
        "windSpeed": 20,
        "visibility": 5,
        "dewpoint": 10,
        "barometricPressure": 1013
      }
    }
  ]
}
```

### GET `/api/health`
Check system health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

## Architecture

### Backend (Flask)
- **app.py**: Main Flask application with REST API
- Loads trained ML models on startup
- Handles image processing and feature extraction
- Makes predictions using fused or single-modality models

### Frontend (Vanilla JavaScript)
- **templates/index.html**: Main dashboard HTML
- **static/css/style.css**: Modern, responsive styling
- **static/js/app.js**: Client-side logic and API communication

### Features
- Image upload with drag-and-drop support
- Real-time weather data input
- Prediction history tracking
- System health monitoring
- Beautiful modal dialogs for various tools

## Extending the Application

### Add Real Data Sources

Modify the Flask app to integrate with:
- **TxDOT Camera API**: For real traffic camera images
- **NOAA Weather API**: For real weather data

Example in `app.py`:
```python
# Replace synthetic data with:
synchronizer = DataSynchronizer()
df = synchronizer.fetch_all_locations()
```

### Custom Styling

Edit `static/css/style.css` to customize colors, fonts, and layout.

### Add More Features

Potential enhancements:
- Real-time map visualization
- Email/SMS alerts for critical conditions
- Database persistence for historical data
- User authentication
- Advanced analytics with charts
- Integration with navigation apps

## Troubleshooting

### "Model loading failed"
- Ensure trained models exist in the `models/` directory
- Check that all required model files are present

### "Connection refused"
- Verify Flask is running on localhost:5000
- Check for port conflicts with `netstat`

### Image upload not working
- Ensure the image file is in a supported format (jpg, png, etc.)
- Check browser console for JavaScript errors

### CORS errors
- Flask-CORS is enabled for all origins
- Check network tab in browser developer tools

## Performance Tips

1. **Image Compression**: Resize large images before upload for faster processing
2. **Batch Analysis**: Use batch-prediction endpoint for analyzing multiple locations simultaneously
3. **Caching**: Consider adding caching for frequently requested location data
4. **Async Processing**: For production, consider using Celery for async model predictions

## Development Mode

For development with auto-reload:

```bash
FLASK_ENV=development python app.py
```

## Production Deployment

For production deployment, use a production WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

This project is part of the Road Hazard Detection System.
