# Real-Time Detection System

This system provides real-time detection of fire, fighting, smoke, and gun activities using your trained YOLOv11 model from an RTSP camera stream.

## Features

- 🔥 **Real-time Detection**: Detects fire, fighting, smoke, and gun activities
- 🎯 **Multi-threaded Processing**: Uses separate threads for frame reading, detection, and display
- 📊 **Live Statistics**: Shows FPS and detection counts in real-time
- 📸 **Screenshot Capability**: Save screenshots of detections
- 📝 **Comprehensive Logging**: Detailed logs for monitoring system status
- 🎨 **Color-coded Bounding Boxes**: Different colors for each detection class

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the real-time detection system:
```bash
python realtime_detection.py
```

### Configuration

The system is pre-configured with:
- **Model Path**: `C:\Users\Abdul Moiz\Desktop\CSMSS_1\runs\detect\train3\weights\best.pt`
- **RTSP URL**: `rtsp://admin:Admin123@192.168.100.2:554/Streaming/Channels/101`
- **Confidence Threshold**: 0.3

### Controls

- **'q'**: Quit the application
- **'s'**: Save a screenshot of the current frame

## System Architecture

The system uses a multi-threaded architecture with three main threads:

1. **Frame Reader Thread**: Continuously reads frames from the RTSP stream
2. **Detection Worker Thread**: Processes frames and runs YOLO detections
3. **Display Worker Thread**: Displays processed frames and handles user input

## Detection Classes

| Class | Color | Description |
|-------|-------|-------------|
| fire | 🔴 Red | Fire detection |
| fighting | 🟢 Green | Fighting/combat detection |
| smoke | ⚫ Gray | Smoke detection |
| gun | 🔵 Blue | Gun/weapon detection |

## Logging

The system provides comprehensive logging:

- **Console Output**: Real-time status updates
- **Log File**: `realtime_detection.log` - Detailed logs with timestamps
- **Detection Alerts**: Immediate logging when detections occur

## Output Files

- **Logs**: `realtime_detection.log`
- **Screenshots**: `screenshots/screenshot_YYYYMMDD_HHMMSS.jpg`

## Performance Optimization

The system includes several optimizations:

- **Queue Management**: Prevents memory overflow by limiting queue sizes
- **Frame Skipping**: Drops old frames when processing is behind
- **FPS Monitoring**: Real-time FPS calculation and display
- **Error Handling**: Robust error handling for network issues

## Troubleshooting

### Connection Issues
- Verify the RTSP URL is correct and accessible
- Check network connectivity to the camera
- Ensure camera credentials are correct

### Performance Issues
- Lower the confidence threshold if needed
- Check system resources (CPU, GPU, memory)
- Consider reducing frame resolution if needed

### Model Issues
- Verify the model file path is correct
- Ensure the model supports the expected classes
- Check if CUDA is available for GPU acceleration

## Customization

### Changing Configuration

Edit the `main()` function in `realtime_detection.py`:

```python
# Configuration
MODEL_PATH = "path/to/your/model.pt"
RTSP_URL = "your_rtsp_url"
CONFIDENCE_THRESHOLD = 0.3
```

### Adding New Classes

To add new detection classes, modify the `RealTimeDetector` class:

```python
self.classes = ['fire', 'fighting', 'smoke', 'gun', 'new_class']
self.class_colors = {
    'fire': (0, 0, 255),
    'fighting': (0, 255, 0),
    'smoke': (128, 128, 128),
    'gun': (255, 0, 0),
    'new_class': (255, 255, 0)  # Yellow
}
```

## System Requirements

- Python 3.8+
- OpenCV 4.8+
- Ultralytics 8.0+
- CUDA-compatible GPU (recommended for better performance)
- Stable network connection for RTSP stream

## Security Notes

- RTSP credentials are stored in plain text in the script
- Consider using environment variables for sensitive information
- Ensure the camera network is properly secured

## Support

For issues or questions:
1. Check the log file for detailed error messages
2. Verify all dependencies are installed correctly
3. Test the RTSP stream with a video player first 