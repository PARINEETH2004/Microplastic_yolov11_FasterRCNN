import os
from ultralytics import YOLO
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainedMicroplasticDetector:
    def __init__(self, model_path=None):
        """Initialize with trained model or fallback to pretrained"""
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path=None):
        """Load the trained model or fallback to pretrained"""
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading trained model from {model_path}")
                self.model = YOLO(model_path)
            else:
                # Check for locally trained model
                local_model = "runs/detect/microplastic_v11/weights/best.pt"
                if os.path.exists(local_model):
                    logger.info(
                        f"Loading local trained model from {local_model}")
                    self.model = YOLO(local_model)
                else:
                    logger.warning(
                        "No trained model found, using pretrained YOLOv11s")
                    self.model = YOLO('yolov11s.pt')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def detect_image(self, image_path, conf_threshold=0.25):
        """Detect microplastics in a single image"""
        try:
            results = self.model(image_path, conf=conf_threshold, iou=0.45)
            return results
        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise

    def detect_video(self, video_path, output_path=None, conf_threshold=0.25):
        """Detect microplastics in video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup output video writer
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run detection
            results = self.model(frame, conf=conf_threshold)

            # Draw results on frame
            annotated_frame = results[0].plot()

            if output_path:
                out.write(annotated_frame)

            frame_count += 1
            if frame_count % 30 == 0:  # Log every 30 frames
                logger.info(f"Processed {frame_count} frames")

        cap.release()
        if output_path:
            out.release()
        logger.info(f"Video processing completed. Total frames: {frame_count}")


def test_detection():
    """Test the detector with sample images"""
    detector = TrainedMicroplasticDetector()

    # Test with a sample image (you can change this path)
    test_image = "test_sample.jpg"  # Replace with your test image path

    if os.path.exists(test_image):
        logger.info(f"Testing detection on {test_image}")
        results = detector.detect_image(test_image)

        # Print results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                logger.info(f"Found {len(boxes)} detections")
                for i, box in enumerate(boxes):
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    logger.info(
                        f"Detection {i+1}: Class {class_id}, Confidence: {confidence:.2f}")
    else:
        logger.warning(f"Test image {test_image} not found")


if __name__ == "__main__":
    print("Microplastic Detection Test")
    print("=" * 30)
    test_detection()
