from ultralytics import YOLO
import shutil
import os
from IPython.display import FileLink  # This library will generate the download link

# 1. Model Setup
# You can use 'yolo11m.pt' or 'yolov8m.pt' or any other model you want
model_name = "rtdetr-l.pt"  
print(f"🚀 Starting training with: {model_name}...")

model = YOLO(model_name)

# 2. Training
try:
    results = model.train(
        data="/kaggle/input/asms-dataset/dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="/kaggle/working/runs/detect",
        name="final_model_train",  # Name of the training folder
        exist_ok=True
    )
except Exception as e:
    print("❌ An error occurred during training!")
    print(e)

# 3. Checking & Zipping
best_weight = "/kaggle/working/runs/detect/final_model_train/weights/best.pt"
output_zip_name = "My_Final_Model.zip"

if os.path.exists(best_weight):
    print("✅ Training finished successfully!")
    print("📦 Compressing files into a Zip archive...")
    
    # Zipping the entire folder so logs and graphs are also included
    shutil.make_archive("My_Final_Model", 'zip', "/kaggle/working/runs/detect/final_model_train")
    
    print("\n" + "="*40)
    print("🎉 DOWNLOAD IS READY! Click the link below:")
    print("="*40 + "\n")
    
    # 4. GENERATE DOWNLOAD LINK
    display(FileLink(output_zip_name))

else:
    print("❌ Error: best.pt file not found. Training might have failed.")