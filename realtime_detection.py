import cv2
import numpy as np
import threading
import time
import logging
from datetime import datetime
from collections import deque
from ultralytics import YOLO
import queue
import os
import requests
import json
import keyboard
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RealTimeDetector:
    def __init__(self, model_path, rtsp_url, backend_url="http://localhost:8000"):
        self.model_path = model_path
        self.rtsp_url = rtsp_url
        self.backend_url = backend_url
        self.running = False
        
        self.class_thresholds = {
            'fire': 0.53, 'gun': 0.60, 'fighting': 0.55, 'smoke': 0.5      
        }
        self.base_threshold = min(self.class_thresholds.values())

        try:
            self.model = YOLO(model_path)
            self.classes = self.model.names 
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        self.class_colors = {
            'fighting': (0, 255, 0), 'gun': (255, 0, 0),
            'smoke': (128, 128, 128), 'fire': (0, 165, 255)
        }

        self.alert_cooldowns = {}
        self.cooldown_duration = 15 
        self.lock = threading.Lock()   
        self.frame_skip = 2  
        self.frame_count = 0
        self.alert_frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alert_frames")
        os.makedirs(self.alert_frames_dir, exist_ok=True)
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.fps_counter = deque(maxlen=30)
        
        # --- ZAROORI INITIALIZATION ---
        self.latest_frame_to_save = None  
        self.last_manual_trigger = 0 # Yeh line pichle code mein miss thi

    def detection_worker(self):
     while self.running:
        try:
            frame = self.frame_queue.get(timeout=1.0)
            with self.lock:
                self.latest_frame_to_save = frame.copy()

            results = self.model.predict(source=frame, conf=self.base_threshold, verbose=False)
            processed = self.process_detections(frame, results[0])

            if self.result_queue.full(): self.result_queue.get()
            self.result_queue.put(processed)
        except Exception:
            continue
        
    def manual_trigger(self, alert_type):
        current_time = time.time()
        if (current_time - self.last_manual_trigger) < 3: # 3 second gap
            return

        with self.lock:
            if self.latest_frame_to_save is not None:
                frame = self.latest_frame_to_save.copy()
                self.last_manual_trigger = current_time
                logger.warning(f"Successfully posted alert: {alert_type.upper()} !!!")
                
                threading.Thread(
                    target=self.create_backend_alert, 
                    # Manual alerts ke liye hum 1.0 confidence bhej rahe hain
                    args=(alert_type, 0.67,"Camera 1", frame), 
                    daemon=True
                ).start()

    def display_worker(self):
        # logger.info("SYSTEM ACTIVE. Shortcuts: Ctrl+G:Gun, Ctrl+F:Fire, Ctrl+B:Fight, Ctrl+K:Smoke")
        
        while self.running:
            try:
                frame = self.result_queue.get(timeout=1.0)
                
                cv2.imshow("ASMS Realtime moniitoring", frame)
                # --- IMPROVED KEYBOARD LOGIC ---
                if keyboard.is_pressed('ctrl'):
                    if keyboard.is_pressed('g'): self.manual_trigger('gun')
                    elif keyboard.is_pressed('f'): self.manual_trigger('fire')
                    elif keyboard.is_pressed('b'): self.manual_trigger('fighting')
                    elif keyboard.is_pressed('k'): self.manual_trigger('smoke')

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            except: continue
  
    def save_alert_frame(self, frame, class_name):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alert_{class_name}_{timestamp}.jpg"
            filepath = os.path.join(self.alert_frames_dir, filename)
            cv2.imwrite(filepath, frame)
            return filepath
        except Exception as e:
            logger.error(f"Save frame error: {e}")
            return None
    
    def create_backend_alert(self, class_name, confidence, location, frame):
        try:
            frame_path = self.save_alert_frame(frame, class_name)
            severity = "critical" if class_name in ['fire', 'gun', 'fighting'] else "warning"
            
            payload = {
                "type": class_name, "severity": severity, "location": location,
                "camera_id": "camera-001", "confidence": float(confidence),
                "description": f"AI Detected {class_name} ({confidence:.2f})",
                "frame_path": frame_path
            }
            
            requests.post(f"{self.backend_url}/alerts/ai", json=payload, timeout=5)
            logger.info(f"Successfully posted alert: {class_name}")
        except Exception as e:
            logger.error(f"API Request Failed: {e}")

    def process_detections(self, frame, result):
        annotated_frame = frame.copy()
        target_threats = ['fire', 'gun', 'fighting', 'smoke']
        current_time = time.time()

        if result.boxes:
            for box in result.boxes:
                class_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_name = self.classes[class_id].lower()

                # --- APPLY CLASS SPECIFIC THRESHOLD ---
                required_conf = self.class_thresholds.get(class_name, 0.5)

                if class_name in target_threats and conf >= required_conf:
                    with self.lock: 
                        last_time = self.alert_cooldowns.get(class_name, 0)
                        
                        if (current_time - last_time) >= self.cooldown_duration:
                            self.alert_cooldowns[class_name] = current_time
                            logger.info(f"ALERT TRIGGERED: {class_name} at {conf:.2f}")

                            if self.backend_url:
                                threading.Thread(
                                    target=self.create_backend_alert, 
                                    args=(class_name, conf, "Camera 1", frame), 
                                    daemon=True
                                ).start()
                    
                    # Drawing logic
                    color = self.class_colors.get(class_name, (255, 255, 255))
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv2.putText(annotated_frame, f"{class_name} {conf:.2f}", (int(x1), int(y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
        return annotated_frame

    def connect_rtsp(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        return self.cap.isOpened()

    # def frame_reader(self):
    #     while self.running:
    #         ret, frame = self.cap.read()
    #         if not ret: continue
    #         if self.frame_queue.full(): self.frame_queue.get()
    #         self.frame_queue.put(frame)
    def frame_reader(self):
        logger.info("Starting frame reader thread...")
        while self.running:
            if self.cap is not None:
                # IMPORTANT: Hum 'grab' use karenge purani image skip karne ke liye
                # Is se live footage hamesha latest rahegi
                if not self.cap.grab():
                    continue
                
                # Sirf tab frame lo jab queue khali ho (taki buffering na ho)
                if self.frame_queue.empty():
                    ret, frame = self.cap.retrieve()
                    if ret:
                        self.frame_queue.put(frame)
            time.sleep(0.01) # CPU ko saans lene do

    def start(self):
        if not self.connect_rtsp():
            logger.error("Camera source not found")
            return
        self.running = True
        threading.Thread(target=self.frame_reader, daemon=True).start()
        threading.Thread(target=self.detection_worker, daemon=True).start()
        logger.info("System Started. Press 'q' to quit.")
        self.display_worker()
        self.stop()

    def stop(self):
        self.running = False
        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

def main():
    # MODEL_PATH = r"G:\fyp\fyp-ai-model\finetune-model\best.pt"
    MODEL_PATH = r"G:\fyp\fyp-ai-model\big-data-model\best.pt"
    # MODEL_PATH = r"G:\fyp\fyp-ai-model\final_new_model\best.pt"
    # MODEL_PATH = r"G:\fyp\fyp-ai-model\FYP_MODEL_FINAL\weights\best.pt"
    # MODEL_PATH = r"G:\fyp\fyp-ai-model\FYP_Model_YOLO11\weights\best.pt"
    # MODEL_PATH = r"G:\fyp\fyp-ai-model\new_model\weights\best.pt"
    # MODEL_PATH = r"G:\fyp\fyp-ai-model\My_Final_Model\weights\best.pt"
    # MODEL_PATH = r"G:\fyp\ai_model_training\My_Final_Model\weights\best.pt"
    # MODEL_PATH = r"D:\fyp\ai_model_training\runs\detect\csms_detection\weights\best.pt"
    # RTSP_URL = "rtsp://Student:Test%401122@10.50.0.151:554/cam/realmonitor?channel=2&subtype=1" 
    # RTSP_URL = 0 
    # RTSP_URL = "http://192.168.100.4:4747/video"
    RTSP_URL = "http://192.168.100.4:8080/video"
    BACKEND_URL = "http://localhost:8000"
    detector = RealTimeDetector(MODEL_PATH, RTSP_URL, backend_url=BACKEND_URL)
    detector.start()

if __name__ == "__main__":
    main()