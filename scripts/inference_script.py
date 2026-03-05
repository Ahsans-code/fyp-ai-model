"""
Inference script for Classroom Safety Monitoring System
Detects safety incidents and sends alerts to the backend API
"""

import cv2
import numpy as np
from ultralytics import YOLO
import requests
import json
import os
import time
from datetime import datetime
import argparse
from pathlib import Path


class CSMSDetector:
    def __init__(self, model_path, backend_url="http://localhost:8000"):
        """
        Initialize the CSMS detector
        
        Args:
            model_path (str): Path to the trained YOLO model
            backend_url (str): URL of the CSMS backend
        """
        self.model = YOLO(model_path)
        self.backend_url = backend_url
        self.alert_threshold = 0.5  # Confidence threshold for alerts
        self.alert_frame_dir = "alert_frames"
        
        # Create directory for alert frames if it doesn't exist
        os.makedirs(self.alert_frame_dir, exist_ok=True)
        
        print(f"CSMS Detector initialized with model: {model_path}")
        print(f"Backend URL: {self.backend_url}")
    
    def send_alert_to_backend(self, detection_result):
        """
        Send alert to the CSMS backend
        
        Args:
            detection_result (dict): Detection result with class, confidence, etc.
        """
        alert_data = {
            "type": detection_result['class_name'],
            "severity": "critical" if detection_result['confidence'] > 0.7 else "warning",
            "location": detection_result.get('location', 'Unknown'),
            "camera_id": detection_result.get('camera_id', 'cam_01'),
            "description": f"Detected {detection_result['class_name']} with {detection_result['confidence']:.2f} confidence",
            "confidence": detection_result['confidence'],
            "frame_path": detection_result.get('frame_path', '')
        }
        
        try:
            response = requests.post(f"{self.backend_url}/alerts/ai", json=alert_data)
            if response.status_code == 200:
                print(f"Alert sent successfully: {alert_data['type']}")
                return True
            else:
                print(f"Failed to send alert. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Error sending alert to backend: {e}")
            return False
    
    def process_frame(self, frame, frame_number=None, location="Classroom A", camera_id="cam_01"):
        """
        Process a single frame and detect safety incidents
        
        Args:
            frame: Image frame to process
            frame_number: Optional frame number for naming
            location: Location identifier
            camera_id: Camera identifier
        """
        # Run inference
        results = self.model(frame)
        
        detections_sent = 0
        
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                
                # Only process detections above the threshold
                if confidence > self.alert_threshold:
                    # Create a copy of the frame with bounding boxes
                    annotated_frame = r.plot()
                    
                    # Generate a unique filename for the alert frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if frame_number is not None:
                        frame_filename = f"alert_{class_name}_{timestamp}_frame{frame_number}.jpg"
                    else:
                        frame_filename = f"alert_{class_name}_{timestamp}.jpg"
                    
                    frame_path = os.path.join(self.alert_frame_dir, frame_filename)
                    cv2.imwrite(frame_path, annotated_frame)
                    
                    # Prepare detection result
                    detection_result = {
                        'class_name': class_name,
                        'confidence': confidence,
                        'location': location,
                        'camera_id': camera_id,
                        'frame_path': frame_path
                    }
                    
                    # Send alert to backend
                    success = self.send_alert_to_backend(detection_result)
                    if success:
                        detections_sent += 1
        
        return detections_sent
    
    def process_video_stream(self, source=0, location="Classroom A", camera_id="cam_01"):
        """
        Process a video stream (webcam or video file)
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            location: Location identifier
            camera_id: Camera identifier
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {source}")
            return
        
        print(f"Processing video stream from {source}...")
        print("Press 'q' to quit")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame")
                break
            
            # Process every frame (or every nth frame to reduce load)
            detections_sent = self.process_frame(frame, frame_count, location, camera_id)
            
            if detections_sent > 0:
                print(f"Sent {detections_sent} detection(s) for frame {frame_count}")
            
            # Display the frame with detections (optional)
            cv2.imshow('CSMS Detection', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("Video stream processing ended")
    
    def process_image(self, image_path, location="Classroom A", camera_id="cam_01"):
        """
        Process a single image file
        
        Args:
            image_path (str): Path to the image file
            location: Location identifier
            camera_id: Camera identifier
        """
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            return 0
        
        print(f"Processing image: {image_path}")
        detections_sent = self.process_frame(frame, location=location, camera_id=camera_id)
        print(f"Processed image. Sent {detections_sent} detection(s)")
        return detections_sent


def main():
    parser = argparse.ArgumentParser(description='CSMS Detection and Alert System')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to the trained YOLO model (.pt file)')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source: 0 for webcam, or path to video/image file')
    parser.add_argument('--backend-url', type=str, default='http://localhost:8000',
                       help='URL of the CSMS backend')
    parser.add_argument('--location', type=str, default='Classroom A',
                       help='Location identifier for alerts')
    parser.add_argument('--camera-id', type=str, default='cam_01',
                       help='Camera identifier for alerts')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for alerts')
    
    args = parser.parse_args()
    
    # Initialize the detector
    detector = CSMSDetector(args.model, args.backend_url)
    detector.alert_threshold = args.threshold
    
    # Determine if source is an image, video, or webcam
    if args.source.isdigit():
        # Webcam
        source = int(args.source)
        detector.process_video_stream(source, args.location, args.camera_id)
    elif Path(args.source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Single image
        detector.process_image(args.source, args.location, args.camera_id)
    else:
        # Video file
        detector.process_video_stream(args.source, args.location, args.camera_id)


if __name__ == "__main__":
    main()