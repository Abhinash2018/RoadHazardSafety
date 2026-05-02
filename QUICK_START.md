# Road Safety Website - Quick Start Guide

## What Was Built

A complete **web application** for your Road Hazard Detection System that allows users to:
1. ✅ Upload road images for hazard analysis
2. ✅ Enter weather conditions to predict road hazards
3. ✅ View real-time predictions with confidence scores
4. ✅ Monitor multiple locations (Austin North, Austin South, San Marcos)
5. ✅ Track prediction history and statistics

## Installation (One-Time Setup)

### Option 1: Windows Users (Easiest)

1. **Double-click** `setup_website.bat` in the project folder
2. Wait for all dependencies to install
3. Installation is complete!

### Option 2: Command Line (Windows/Mac/Linux)

```bash
# Navigate to the project folder
cd path\to\RoadSafety

# Install dependencies
pip install -r requirements.txt
```

## Running the Website

### Option 1: Windows Users

**Double-click** `run_website.bat` - The website will automatically open at `http://localhost:5000`

### Option 2: Command Line

```bash
python app.py
```

Then open your browser and go to: **http://localhost:5000**

## Website Features at a Glance

### 📍 Location Cards
- Shows all monitored locations along the I-35 corridor
- Click any location to analyze its weather conditions

### 📸 Image Upload Tool
- Drag & drop road images or click to select
- Analyzes visual features + optional weather data
- Shows prediction with confidence scores

### 🌤️ Weather Analysis
- Select a location
- Enter weather conditions (temperature, humidity, wind, etc.)
- Get instant hazard prediction for that location

### 📊 Prediction History
- View all recent predictions in real-time
- Shows location, time, hazard level, and confidence

### 🔧 Analysis Tools
- **Upload Image**: Analyze a single road photo
- **Weather Analysis**: Predict hazards based on weather data
- **Historical Data**: View statistics and trends

## Testing the Website

### Quick Test

1. Launch the website (`run_website.bat`)
2. Click **"Weather Analysis"** button
3. Select "Austin North" location
4. Fill in weather data:
   - Temperature: 5°C
   - Humidity: 80%
   - Wind Speed: 30 km/h
   - Visibility: 2 km
   - Dewpoint: 2°C
   - Pressure: 1010 hPa
5. Click **"Predict Hazard Level"**
6. You should see a prediction (likely "icy" or "storm_risk" for this weather)

### Testing with an Image

1. Find any road image on your computer
2. Click **"Upload Image"** button
3. Drag and drop the image (or click to browse)
4. Click **"Analyze"**
5. The system will analyze the road conditions from the image

## File Structure

```
RoadSafety/
├── app.py                      # Flask web server
├── requirements.txt            # Python dependencies (updated with Flask)
├── WEBSITE_README.md           # Full documentation
├── QUICK_START.md             # This file
├── setup_website.bat          # Setup script (Windows)
├── setup_website.sh           # Setup script (Mac/Linux)
├── run_website.bat            # Run script (Windows)
├── templates/
│   └── index.html             # Main dashboard (HTML)
├── static/
│   ├── css/
│   │   └── style.css          # Dashboard styling (CSS)
│   └── js/
│       └── app.js             # Dashboard logic (JavaScript)
└── [other project files]
```

## API Endpoints Available

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main dashboard page |
| `/api/locations` | GET | Get all monitored locations |
| `/api/prediction` | POST | Make prediction from image/weather |
| `/api/batch-prediction` | POST | Predict for multiple locations |
| `/api/health` | GET | Check system status |

## Troubleshooting

### Port Already in Use
If port 5000 is already in use:
```bash
python app.py
```
The app will try another port automatically, or modify `app.py` line: `port=5000`

### Models Not Loading
- Ensure you have trained models in the `models/` directory
- Check that all required model files are present
- The app will show "Models Loading..." status if they're not ready

### JavaScript Errors
- Open browser developer tools (F12)
- Check the Console tab for errors
- Ensure all files were created correctly (app.py, templates/index.html, static/css/style.css, static/js/app.js)

### Can't Connect to Server
- Check that Flask is running (you should see "Running on http://localhost:5000")
- Try refreshing the browser
- Check if port 5000 is blocked by firewall

## How It Works

### Backend (Flask - app.py)
- Loads your trained ML models on startup
- Provides REST API endpoints
- Processes image uploads
- Makes hazard predictions
- Returns results as JSON

### Frontend (HTML/CSS/JavaScript)
- Beautiful, responsive dashboard interface
- Handles user interactions (uploads, forms, etc.)
- Communicates with backend via API
- Displays predictions in real-time
- Shows system status and prediction history

### Model Integration
- Your existing `RoadHazardClassifier` models are loaded
- `VisualFeatureExtractor` processes images
- Predictions can be:
  - **Image-only**: Visual features from road photos
  - **Weather-only**: Hazard based on weather conditions
  - **Fused**: Combined image + weather for best accuracy

## Next Steps

### To Deploy to Production
See the "Production Deployment" section in WEBSITE_README.md

### To Add More Features
- Add real TxDOT camera API integration
- Add real NOAA weather API integration
- Add database for historical data
- Add user authentication
- Add email alerts for critical conditions
- Add map visualization with markers

### To Customize
- Modify colors in `static/css/style.css`
- Change layout in `templates/index.html`
- Update API logic in `app.py`

## Support Files

For detailed information, see:
- **WEBSITE_README.md** - Full API documentation and features
- **app.py** - Backend implementation with detailed comments
- **static/js/app.js** - Frontend logic with detailed comments
- **static/css/style.css** - Styling with detailed comments

## Command Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Run the website
python app.py

# Run with debug enabled
FLASK_ENV=development python app.py

# Check if Flask is installed
pip show flask

# Check if ports are in use (Windows)
netstat -ano | findstr :5000

# Kill process on port 5000 (Windows)
taskkill /PID <PID> /F
```

## Questions?

1. Check WEBSITE_README.md for detailed API documentation
2. Look at the comments in app.py for backend logic
3. Check browser console (F12) for frontend errors
4. Verify your ML models are properly trained and in the models/ directory

---

**You're all set!** 🎉 Run `python app.py` and visit http://localhost:5000
