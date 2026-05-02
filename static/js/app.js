/* ============================================
   Road Safety Dashboard - JavaScript
   ============================================ */

const API_BASE = 'http://localhost:5000/api';
let locations = [];
let predictions = [];
let selectedImage = null;

/* ============================================
   Initialization
   ============================================ */

document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
});

function initializeDashboard() {
    // Load locations
    loadLocations();
    
    // Check system health
    checkHealth();
    
    // Setup event listeners
    setupEventListeners();
    
    // Auto-refresh health check every 30 seconds
    setInterval(checkHealth, 30000);
}

function setupEventListeners() {
    // Upload area drag and drop
    const uploadArea = document.getElementById('uploadArea');
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        uploadArea.addEventListener('click', () => {
            document.getElementById('imageInput').click();
        });
    }
    
    // Image input change
    const imageInput = document.getElementById('imageInput');
    if (imageInput) {
        imageInput.addEventListener('change', handleImageSelect);
    }
    
    // Weather data checkbox
    const includeWeatherCheckbox = document.getElementById('includeWeatherData');
    if (includeWeatherCheckbox) {
        includeWeatherCheckbox.addEventListener('change', (e) => {
            const form = document.getElementById('weatherDataForm');
            if (form) form.style.display = e.target.checked ? 'block' : 'none';
        });
    }
    
    // Close modals when clicking outside
    window.addEventListener('click', (e) => {
        const modals = document.querySelectorAll('.modal');
        modals.forEach(modal => {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    });
}

/* ============================================
   Location Management
   ============================================ */

async function loadLocations() {
    try {
        const response = await fetch(`${API_BASE}/locations`);
        if (!response.ok) throw new Error('Failed to load locations');
        
        locations = await response.json();
        renderLocations();
        populateLocationSelects();
    } catch (error) {
        console.error('Error loading locations:', error);
        showNotification('Failed to load locations', 'error');
    }
}

function renderLocations() {
    const grid = document.getElementById('locationsGrid');
    if (!grid) return;
    
    grid.innerHTML = locations.map(location => `
        <div class="location-card" onclick="simulateLocationPrediction('${location.id}')">
            <h3><i class="fas fa-map-pin"></i> ${location.name}</h3>
            <p>Lat: ${location.latitude.toFixed(4)}<br>Lon: ${location.longitude.toFixed(4)}</p>
            <span class="location-badge badge-safe">Not Analyzed</span>
            <p style="font-size: 12px; margin-top: 10px; color: #999;">Click to analyze weather</p>
        </div>
    `).join('');
}

function populateLocationSelects() {
    const selects = document.querySelectorAll('#weatherLocation');
    selects.forEach(select => {
        select.innerHTML = locations.map(loc => 
            `<option value="${loc.id}">${loc.name}</option>`
        ).join('');
    });
}

function simulateLocationPrediction(locationId) {
    const location = locations.find(l => l.id === locationId);
    if (location) {
        openWeatherAnalysis();
        // Pre-fill with location
        const select = document.getElementById('weatherLocation');
        if (select) select.value = locationId;
    }
}

/* ============================================
   Image Upload & Prediction
   ============================================ */

function handleDragOver(e) {
    e.preventDefault();
    e.target.style.borderColor = '#2563eb';
    e.target.style.background = 'rgba(37, 99, 235, 0.1)';
}

function handleDragLeave(e) {
    e.target.style.borderColor = '#e5e7eb';
    e.target.style.background = '#f9fafb';
}

function handleDrop(e) {
    e.preventDefault();
    e.target.style.borderColor = '#e5e7eb';
    e.target.style.background = '#f9fafb';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleImageFile(files[0]);
    }
}

function handleImageSelect(e) {
    if (e.target.files.length > 0) {
        handleImageFile(e.target.files[0]);
    }
}

function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('Please select a valid image file', 'error');
        return;
    }
    
    selectedImage = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        const preview = document.getElementById('imagePreview');
        const img = document.getElementById('previewImg');
        if (preview && img) {
            img.src = e.target.result;
            preview.style.display = 'block';
        }
    };
    reader.readAsDataURL(file);
}

async function submitImagePrediction() {
    const resultDiv = document.getElementById('predictionResult');
    
    if (!selectedImage && !document.getElementById('includeWeatherData').checked) {
        showNotification('Please upload an image or include weather data', 'error');
        return;
    }
    
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '<p><span class="loading"></span> Analyzing...</p>';
    
    try {
        const formData = new FormData();
        
        if (selectedImage) {
            formData.append('image', selectedImage);
        }
        
        if (document.getElementById('includeWeatherData').checked) {
            const weatherData = {
                temperature: parseFloat(document.getElementById('weatherTemp').value) || 20,
                relativeHumidity: parseFloat(document.getElementById('weatherHumidity').value) || 60,
                windSpeed: parseFloat(document.getElementById('weatherWindSpeed').value) || 10,
                visibility: parseFloat(document.getElementById('weatherVisibility').value) || 10,
                dewpoint: parseFloat(document.getElementById('weatherDewpoint').value) || 10,
                barometricPressure: parseFloat(document.getElementById('weatherPressure').value) || 1013
            };
            formData.append('weather_data', JSON.stringify(weatherData));
        }
        
        const response = await fetch(`${API_BASE}/prediction`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        displayImagePredictionResult(data);
        
        if (data.success) {
            addPrediction(data);
        }
    } catch (error) {
        console.error('Error:', error);
        resultDiv.innerHTML = `<div class="error"><strong>Error:</strong> ${error.message}</div>`;
    }
}

function displayImagePredictionResult(data) {
    const resultDiv = document.getElementById('predictionResult');
    
    if (data.success) {
        const hazardClass = data.hazard_class;
        const hazardLabels = ['safe', 'wet', 'snowy', 'icy', 'storm_risk'];
        const hazardColors = {
            'safe': 'badge-safe',
            'wet': 'badge-wet',
            'snowy': 'badge-snowy',
            'icy': 'badge-icy',
            'storm_risk': 'badge-storm'
        };
        
        resultDiv.className = 'prediction-result success';
        resultDiv.innerHTML = `
            <h3><i class="fas fa-check-circle"></i> Analysis Complete</h3>
            <div class="result-grid">
                <div class="result-item">
                    <span class="result-label">Road Condition:</span>
                    <span class="location-badge ${hazardColors[data.prediction]}">${data.prediction.toUpperCase()}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Confidence:</span>
                    <span class="result-value">${data.confidence}%</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Features Used:</span>
                    <span class="result-value">${data.features_used.join(', ')}</span>
                </div>
            </div>
            ${data.confidence_scores ? `
                <div style="margin-top: 15px;">
                    <h4>Confidence Scores:</h4>
                    <div class="result-grid">
                        ${Object.entries(data.confidence_scores).map(([label, score]) => `
                            <div class="result-item">
                                <span class="result-label">${label.toUpperCase()}:</span>
                                <span class="result-value">${score}%</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            ` : ''}
        `;
    } else {
        resultDiv.className = 'prediction-result error';
        resultDiv.innerHTML = `<strong>Error:</strong> ${data.error || 'Failed to process image'}`;
    }
}

/* ============================================
   Weather Prediction
   ============================================ */

async function submitWeatherPrediction() {
    const location = document.getElementById('weatherLocation').value;
    const weatherData = {
        temperature: parseFloat(document.getElementById('wTemp').value) || 20,
        relativeHumidity: parseFloat(document.getElementById('wHumidity').value) || 60,
        windSpeed: parseFloat(document.getElementById('wWindSpeed').value) || 10,
        visibility: parseFloat(document.getElementById('wVisibility').value) || 10,
        dewpoint: parseFloat(document.getElementById('wDewpoint').value) || 10,
        barometricPressure: parseFloat(document.getElementById('wPressure').value) || 1013
    };
    
    const resultDiv = document.getElementById('weatherPredictionResult');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '<p><span class="loading"></span> Predicting...</p>';
    
    try {
        const response = await fetch(`${API_BASE}/batch-prediction`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                items: [{
                    location: location,
                    weather: weatherData
                }]
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.predictions.length > 0) {
            const pred = data.predictions[0];
            const hazardColors = {
                'safe': 'badge-safe',
                'wet': 'badge-wet',
                'snowy': 'badge-snowy',
                'icy': 'badge-icy',
                'storm_risk': 'badge-storm'
            };
            
            resultDiv.className = 'prediction-result success';
            resultDiv.innerHTML = `
                <h3><i class="fas fa-cloud"></i> Weather-Based Prediction</h3>
                <div class="result-grid">
                    <div class="result-item">
                        <span class="result-label">Location:</span>
                        <span class="result-value">${location}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Predicted Condition:</span>
                        <span class="location-badge ${hazardColors[pred.prediction]}">${pred.prediction.toUpperCase()}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Confidence:</span>
                        <span class="result-value">${pred.confidence}%</span>
                    </div>
                </div>
                <div style="margin-top: 15px; padding: 10px; background: white; border-radius: 4px; font-size: 13px;">
                    <p><strong>Weather Conditions:</strong></p>
                    <p>Temperature: ${weatherData.temperature}°C | Humidity: ${weatherData.relativeHumidity}%</p>
                    <p>Wind: ${weatherData.windSpeed} km/h | Visibility: ${weatherData.visibility} km</p>
                </div>
            `;
            
            addPrediction({
                prediction: pred.prediction,
                confidence: pred.confidence,
                location: location,
                features_used: ['weather']
            });
        }
    } catch (error) {
        console.error('Error:', error);
        resultDiv.className = 'prediction-result error';
        resultDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
    }
}

/* ============================================
   Predictions Management
   ============================================ */

function addPrediction(predictionData) {
    const prediction = {
        id: Date.now(),
        location: predictionData.location || 'Direct Analysis',
        hazard_class: predictionData.hazard_class || predictionData.prediction,
        confidence: predictionData.confidence,
        timestamp: new Date().toLocaleTimeString(),
        features: predictionData.features_used
    };
    
    predictions.unshift(prediction);
    if (predictions.length > 10) predictions.pop(); // Keep last 10
    
    renderPredictions();
}

function renderPredictions() {
    const list = document.getElementById('predictionsList');
    if (!list) return;
    
    if (predictions.length === 0) {
        list.innerHTML = '<p class="empty-state">No predictions yet. Upload an image or enter weather data.</p>';
        return;
    }
    
    const hazardColors = {
        'safe': 'badge-safe',
        'wet': 'badge-wet',
        'snowy': 'badge-snowy',
        'icy': 'badge-icy',
        'storm_risk': 'badge-storm'
    };
    
    list.innerHTML = predictions.map(pred => {
        const hazardLabel = typeof pred.hazard_class === 'string' ? 
            pred.hazard_class : 
            ['safe', 'wet', 'snowy', 'icy', 'storm_risk'][pred.hazard_class] || 'unknown';
        
        return `
            <div class="prediction-item">
                <div class="prediction-info">
                    <h4>${pred.location}</h4>
                    <p><strong>Time:</strong> ${pred.timestamp}</p>
                    <p><strong>Features:</strong> ${pred.features ? pred.features.join(', ') : 'weather'}</p>
                </div>
                <div class="prediction-badge">
                    <span class="prediction-class ${hazardColors[hazardLabel]}">${hazardLabel}</span>
                    <span class="confidence">${pred.confidence}%</span>
                </div>
            </div>
        `;
    }).join('');
}

/* ============================================
   System Health & Status
   ============================================ */

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        
        if (statusDot && statusText) {
            if (data.status === 'healthy' && data.model_loaded) {
                statusDot.classList.add('connected');
                statusText.textContent = 'System Connected & Ready';
            } else {
                statusDot.classList.remove('connected');
                statusText.textContent = data.model_loaded ? 
                    'System Connected' : 'Models Loading...';
            }
        }
    } catch (error) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        if (statusDot && statusText) {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Connection Error';
        }
    }
}

/* ============================================
   Modal Management
   ============================================ */

function openImageUpload() {
    document.getElementById('imageUploadModal').style.display = 'block';
}

function openWeatherAnalysis() {
    document.getElementById('weatherModal').style.display = 'block';
}

function openHistoricalData() {
    updateStatistics();
    document.getElementById('historicalModal').style.display = 'block';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

/* ============================================
   Statistics
   ============================================ */

function updateStatistics() {
    const totalPredictions = predictions.length;
    const safePredictions = predictions.filter(p => {
        const hazardLabel = typeof p.hazard_class === 'string' ? p.hazard_class : ['safe', 'wet', 'snowy', 'icy', 'storm_risk'][p.hazard_class];
        return hazardLabel === 'safe';
    }).length;
    const criticalAlerts = predictions.filter(p => {
        const hazardLabel = typeof p.hazard_class === 'string' ? p.hazard_class : ['safe', 'wet', 'snowy', 'icy', 'storm_risk'][p.hazard_class];
        return hazardLabel === 'storm_risk' || hazardLabel === 'icy';
    }).length;
    const avgConfidence = predictions.length > 0 ? 
        Math.round(predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length) : 
        0;
    
    document.getElementById('totalPredictions').textContent = totalPredictions;
    document.getElementById('safePredictions').textContent = safePredictions;
    document.getElementById('criticalAlerts').textContent = criticalAlerts;
    document.getElementById('avgConfidence').textContent = avgConfidence + '%';
}

/* ============================================
   Notifications
   ============================================ */

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'error' ? '#ef4444' : type === 'success' ? '#10b981' : '#3b82f6'};
        color: white;
        border-radius: 8px;
        box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2);
        z-index: 2000;
        animation: slideInRight 0.3s ease-out;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animation keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
