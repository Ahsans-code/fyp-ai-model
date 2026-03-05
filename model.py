from ultralytics import YOLO
import os


model= YOLO("yolo11n.pt") 
model.train(data="C:/Users/ASUS/Desktop/CSMSS_1/data.yaml", imgsz=640, batch=16, epochs=50, workers=0, device=0)
