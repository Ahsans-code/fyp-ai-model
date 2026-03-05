# CSMS AI Model - Backend Integration

This document explains how to integrate the AI model with the CSMS backend system to create real-time alerts.

## Overview

The AI model now has the capability to automatically create alerts in the backend system when it detects security threats (fighting, fire, gun, smoke). The system includes a cooldown mechanism to prevent duplicate alerts for the same detection type.

## Features

1. **Real-time Detection**: Detects fighting, fire, gun, and smoke in real-time video streams
2. **Backend Integration**: Automatically creates alerts in the CSMS backend system
3. **Cooldown System**: Prevents duplicate alerts (30-second cooldown per detection type)
4. **Confidence Thresholds**: 
   - Fighting: Requires 0.55+ confidence
   - Fire, Gun, Smoke: Use default threshold (0.3)
5. **Asynchronous Alert Creation**: Alert creation runs in background threads to avoid blocking detection

## Setup Instructions

### 1. Start the Backend Server

First, make sure your CSMS backend is running:

```bash
cd "csms backend"
# Activate your virtual environment if needed
python main.py
```

The backend should be running on `http://localhost:8000`

### 2. Get Authentication Token

Run the authentication helper script to get a token:

```bash
cd "csms ai model"
python get_auth_token.py
```

This will:
- Register a test user (if not already exists)
- Get an authentication token
- Display the token to use in the main script

### 3. Update the Main Script

Edit `realtime_detection.py` and update the `AUTH_TOKEN` variable in the `main()` function:

```python
def main():
    # Configuration
    MODEL_PATH = r"C:\Users\Abdul Moiz\Desktop\csms all\csms ai model\runs\detect\train3\weights\best.pt"
    RTSP_URL = "rtsp://admin:Admin123@192.168.100.2:554/Streaming/Channels/101"
    CONFIDENCE_THRESHOLD = 0.3
    
    # Backend configuration
    BACKEND_URL = "http://localhost:8000"
    AUTH_TOKEN = "your_token_here"  # Replace with the token from get_auth_token.py
    
    # ... rest of the function
```

### 4. Run the Detection System

```bash
python realtime_detection.py
```

## How It Works

### Detection Process

1. **Frame Processing**: The system continuously processes video frames
2. **AI Detection**: YOLO model detects objects with confidence scores
3. **Confidence Filtering**: 
   - Fighting detections must have ≥0.55 confidence
   - Other detections use the default threshold (0.3)
4. **Alert Creation**: When a detection passes the confidence threshold:
   - Checks if enough time has passed since the last alert for this type (30-second cooldown)
   - Creates an alert in the backend system
   - Updates the cooldown timestamp

### Alert Data Structure

Each alert sent to the backend includes:

```json
{
  "type": "fighting|fire|gun|smoke",
  "severity": "warning|critical",
  "location": "Camera 1",
  "camera_id": "camera-001",
  "description": "Fighting detected with 0.85 confidence",
  "confidence": 0.85
}
```

### Cooldown System

- **Purpose**: Prevents spam alerts when the same object is detected continuously
- **Duration**: 30 seconds per detection type
- **Behavior**: Only one alert per detection type every 30 seconds
- **Example**: If fighting is detected at 10:00:00, the next fighting alert can only be created at 10:00:30 or later

## Configuration Options

### Alert Cooldown Duration

To change the cooldown duration, modify the `cooldown_duration` variable in the `RealTimeDetector` class:

```python
self.cooldown_duration = 30  # Change to desired seconds
```

### Severity Thresholds

To change how severity is determined, modify the `create_backend_alert` method:

```python
# Current: confidence > 0.7 = critical, otherwise warning
severity = "critical" if confidence > 0.7 else "warning"

# You can customize this logic
if confidence > 0.8:
    severity = "critical"
elif confidence > 0.6:
    severity = "warning"
else:
    severity = "info"
```

### Backend URL

To change the backend URL, modify the `BACKEND_URL` variable in the `main()` function:

```python
BACKEND_URL = "http://your-backend-url:port"
```

## Troubleshooting

### Common Issues

1. **"No auth token provided"**: 
   - Run `get_auth_token.py` to get a valid token
   - Update the `AUTH_TOKEN` variable in `realtime_detection.py`

2. **"Failed to create alert"**:
   - Check if the backend server is running
   - Verify the backend URL is correct
   - Check if the token is valid and not expired

3. **"Alert is in cooldown"**:
   - This is normal behavior to prevent spam
   - Wait 30 seconds or reduce the cooldown duration

4. **Connection errors**:
   - Check network connectivity
   - Verify backend server is accessible
   - Check firewall settings

### Logs

The system logs all activities to `realtime_detection.log`. Check this file for detailed information about:

- Detection events
- Alert creation attempts
- Backend communication
- Errors and warnings

## Testing

To test the integration:

1. Start the backend server
2. Get an authentication token
3. Update the main script with the token
4. Run the detection system
5. Trigger detections (e.g., show fighting videos)
6. Check the frontend to see alerts appear in real-time

The alerts should appear in the CSMS frontend dashboard and be broadcast via WebSocket to all connected clients. 