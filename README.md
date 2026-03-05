# Classroom Safety Monitoring System - AI Model Training & Deployment

This directory contains the complete AI model training and deployment code for the Classroom Safety Monitoring System (CSMS).

## Directory Structure

```
ai_model_training/
├── yolo11n.pt                 # Trained YOLO11 model for threat detection
├── realtime_detection.py      # Main real-time camera detection system
├── camera_processor.py        # Alternative camera processing script
├── get_auth_token.py          # Authentication helper for backend
├── model.py                   # Model utilities
├── predict.py                 # Prediction utilities
├── test_model.py              # Model testing utilities
├── alert_frames/              # Directory for saved alert frames
├── screenshots/               # Directory for manual screenshots
├── scripts/
│   ├── train_model.py         # Training script
│   └── inference_script.py    # Standalone inference script
├── dataset/                   # Training dataset (if needed)
├── trained_models/            # Additional trained models
├── training_logs/             # Training logs and metrics
├── requirements.txt           # Python dependencies
├── README_realtime.md         # Real-time detection documentation
├── README_INTEGRATION.md      # Backend integration documentation
└── README.md                 # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Directories

Create necessary directories:
```bash
mkdir alert_frames
mkdir screenshots
```

## Complete System Testing

### Step 1: Start Backend Server
```bash
cd ../backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2: Register User and Get Token (Optional but Recommended)
```bash
cd ../ai_model_training
python get_auth_token.py
```

### Step 3: Run Real-time Detection
```bash
python realtime_detection.py
```

### Step 4: Start Frontend (in a new terminal)
```bash
cd ../frontend
npm start
```

### Step 5: Enable Audio Alerts
- Open the dashboard in your browser
- Click the "Enable Audio Alerts" button
- The system is now monitoring for threats

## How to Test Each Component

### Test Backend Connection
1. Ensure backend is running on `http://localhost:8000`
2. Visit `http://localhost:8000/docs` to see API documentation
3. Test the `/alerts/ai` endpoint manually if needed

### Test Camera Connection
1. Ensure your RTSP camera is accessible
2. The default RTSP URL is: `rtsp://admin:Admin123@192.168.100.2:554/Streaming/Channels/101`
3. Update in `realtime_detection.py` if needed

### Test AI Detection
1. The system will detect:
   - **Fire** (red bounding box)
   - **Fighting** (green bounding box, requires 0.55+ confidence)
   - **Smoke** (gray bounding box)
   - **Gun** (blue bounding box)

### Test Alert System
1. When a threat is detected, an alert should appear in the dashboard
2. Audio alerts should play based on severity
3. Alert frames should be saved in the `alert_frames/` directory
4. WebSocket should broadcast alerts in real-time

### Test Cooldown System
- The system has a 30-second cooldown per threat type
- This prevents spam alerts when the same threat is continuously detected

## Configuration Options

### Update RTSP URL
Edit `realtime_detection.py`:
```python
RTSP_URL = "your_rtsp_url_here"
```

### Update Confidence Thresholds
Edit `realtime_detection.py`:
```python
CONFIDENCE_THRESHOLD = 0.3  # Default for most classes
# Fighting has a higher threshold of 0.55 in the code
```

### Update Backend URL
Edit `realtime_detection.py`:
```python
BACKEND_URL = "http://your-backend-url:port"
```

## Troubleshooting

### Common Issues

1. **"Failed to connect to RTSP stream"**:
   - Verify the RTSP URL is correct
   - Check network connectivity to the camera
   - Ensure camera credentials are correct

2. **"Failed to create alert"**:
   - Check if backend server is running
   - Verify backend URL is correct
   - Check network connectivity

3. **"No audio alerts"**:
   - Click "Enable Audio Alerts" button in dashboard
   - Check browser audio settings
   - Verify WebSocket connection

4. **"Model not loading"**:
   - Verify `yolo11n.pt` exists in the correct location
   - Check that ultralytics is properly installed

### Logs
- Detection logs: `realtime_detection.log`
- Backend logs: Console output from uvicorn
- Browser console: Frontend WebSocket connection status

## System Requirements

- Python 3.8+
- OpenCV 4.8+
- Ultralytics 8.0+
- CUDA-compatible GPU (recommended for better performance)
- Stable network connection for RTSP stream
- At least 4GB RAM for model processing

## Security Notes

- RTSP credentials are stored in plain text in the script
- Consider using environment variables for sensitive information
- Ensure the camera network is properly secured

## Performance Optimization

The system includes several optimizations:
- Multi-threaded processing for better performance
- Queue management to prevent memory overflow
- Frame skipping when processing is behind
- FPS monitoring for real-time performance tracking