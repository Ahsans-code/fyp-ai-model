import os

import cv2
from ultralytics import YOLO

# Step 1: Load trained model
model = YOLO(r"C:\Users\Abdul Moiz\Desktop\CSMSS_1\runs\detect\train3\weights\best.pt")  # ya koi aur .pt file

# Step 2: Set path of the input image
image_path = r'C:\Users\Abdul Moiz\Desktop\CSMSS_1\test\images\armas-840-jpg_jpg.rf.8e7cf81818ac7852bcc8d44cec7d0e87.jpg'  # ← yahan apni image ka full path dein

# Step 3: Predict using YOLOv8
results = model.predict(source=image_path, save=False, save_txt=False, conf=0.3)

# Prepare save directory
output_dir = 'predictions'
os.makedirs(output_dir, exist_ok=True)

# Generate auto-incrementing filename
existing_files = [f for f in os.listdir(output_dir) if f.startswith("prediction") and f.endswith(".jpg")]
numbers = [int(f.replace("prediction", "").replace(".jpg", "")) for f in existing_files if f.replace("prediction", "").replace(".jpg", "").isdigit()]
next_num = max(numbers, default=0) + 1
output_filename = f"prediction{next_num}.jpg"
output_path = os.path.join(output_dir, output_filename)

# Save predicted image
annotated_frame = results[0].plot()
cv2.imwrite(output_path, annotated_frame)

# Confirmation
print(f"✅ Prediction saved at: {output_path}")

# Print detected classes
print("\n🔍 Detected objects:")
if results[0].boxes:
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        print(f"→ {class_name}")
else:
    print("⚠️ No objects detected. means null object")