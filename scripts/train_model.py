"""
Training script for Classroom Safety Monitoring System AI Model
Trains a YOLOv8 model to detect fire, fighting, smoke, and gun incidents
"""

import os
import sys
from ultralytics import YOLO
import argparse


def train_model(data_yaml_path, epochs=100, img_size=640, batch_size=16, model_type='yolov8n'):
    """
    Train the YOLO model for safety incident detection
    
    Args:
        data_yaml_path (str): Path to the dataset YAML file
        epochs (int): Number of training epochs
        img_size (int): Image size for training
        batch_size (int): Batch size for training
        model_type (str): YOLO model type (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
    """
    
    print(f"Starting training with {model_type} model...")
    print(f"Dataset: {data_yaml_path}")
    print(f"Epochs: {epochs}, Image Size: {img_size}, Batch Size: {batch_size}")
    
    # Load a pre-trained model
    model = YOLO(f'{model_type}.pt')
    
    # Start training
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='csms_detection',
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache=False,     # Set to True if you have enough RAM to cache dataset
        device=0 if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    )
    
    print("Training completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    return results


def validate_model(model_path, data_yaml_path):
    """
    Validate the trained model
    
    Args:
        model_path (str): Path to the trained model
        data_yaml_path (str): Path to the dataset YAML file for validation
    """
    
    print(f"Validating model: {model_path}")
    
    # Load the trained model
    model = YOLO(model_path)
    
    # Validate the model
    metrics = model.val(data=data_yaml_path)
    
    print("Validation Results:")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model for CSMS')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--model-type', type=str, default='yolov8n', 
                       choices=['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'],
                       help='YOLO model type')
    parser.add_argument('--validate', action='store_true', help='Run validation after training')
    
    args = parser.parse_args()
    
    # Import torch here to check GPU availability
    import torch
    
    # Train the model
    results = train_model(
        data_yaml_path=args.data,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        model_type=args.model_type
    )
    
    # Optionally validate the model
    if args.validate:
        best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            validate_model(best_model_path, args.data)
        else:
            print(f"Best model not found at {best_model_path}")