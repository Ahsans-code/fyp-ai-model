import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model():
    """Test the trained model to verify it's working correctly"""
    
    MODEL_PATH = r"C:\Users\Abdul Moiz\Desktop\CSMSS_1\runs\detect\train3\weights\best.pt"
    
    logger.info("Testing YOLO model...")
    
    try:
        # Load model
        model = YOLO(MODEL_PATH)
        logger.info("Model loaded successfully")
        
        # Check model classes
        if hasattr(model, 'names'):
            logger.info(f"Model classes: {model.names}")
            logger.info(f"Number of classes: {len(model.names)}")
            
            # Print each class with its ID
            for class_id, class_name in model.names.items():
                logger.info(f"  Class {class_id}: {class_name}")
        
        # Test with a dummy image
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        logger.info("Running test prediction...")
        
        results = model.predict(source=test_image, conf=0.1, save=False, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            logger.info(f"Test prediction successful")
            logger.info(f"Number of detections: {len(result.boxes) if result.boxes else 0}")
            
            if result.boxes and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    logger.info(f"  Detection {i}: class_id={class_id}, confidence={confidence:.2f}")
        
        # Test with a real image if available
        test_image_path = "test/images/1026_jpg.rf.c1e7f743403766ae167c27e275b4dca1.jpg"
        try:
            logger.info(f"Testing with real image: {test_image_path}")
            results = model.predict(source=test_image_path, conf=0.3, save=False, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                logger.info(f"Real image prediction successful")
                logger.info(f"Number of detections: {len(result.boxes) if result.boxes else 0}")
                
                if result.boxes and len(result.boxes) > 0:
                    for i, box in enumerate(result.boxes):
                        class_id = int(box.cls[0].cpu().numpy())
                        confidence = box.conf[0].cpu().numpy()
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_name = model.names.get(class_id, f"Unknown_{class_id}")
                        logger.info(f"  Detection {i}: {class_name} (confidence: {confidence:.2f}) at ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
        
        except Exception as e:
            logger.warning(f"Could not test with real image: {e}")
        
        logger.info("Model test completed successfully")
        
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_model() 