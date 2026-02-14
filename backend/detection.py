import cv2
import numpy as np
from ultralytics import YOLO
import random
import os
from typing import List, Dict, Any, Tuple
from config import Config
import logging

logger = logging.getLogger(__name__)


class MicroplasticDetector:
    def __init__(self):
        self.config = Config()
        self.yolo_model = None
        self.faster_rcnn_detector = None
        self._load_models()

    def _load_models(self):
        """Load both YOLOv11 and Faster R-CNN models"""
        # Load YOLO model
        self._load_yolo_model()

        # Load Faster R-CNN model
        self._load_faster_rcnn_model()

    def _load_yolo_model(self):
        """Load the YOLOv11 model"""
        try:
            # Check for trained model first
            trained_model_path = os.path.join(os.path.dirname(
                __file__), 'runs', 'detect', 'microplastic_v11', 'weights', 'best.pt')
            if os.path.exists(trained_model_path):
                logger.info(
                    f"Loading trained YOLOv11 microplastic model from {trained_model_path}")
                self.yolo_model = YOLO(trained_model_path)
            elif os.path.exists(self.config.MODEL_PATH):
                logger.info(
                    f"Loading custom YOLOv11 model from {self.config.MODEL_PATH}")
                self.yolo_model = YOLO(self.config.MODEL_PATH)
            else:
                logger.warning(
                    f"YOLOv11 model file not found at {self.config.MODEL_PATH}")
                logger.info("Using YOLOv11s pretrained model as fallback")
                # Download and load pretrained model
                self.yolo_model = YOLO('yolov11s.pt')
            logger.info("YOLOv11 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLOv11 model: {e}")
            # Try to download the model
            try:
                logger.info("Attempting to download YOLOv11s model...")
                self.yolo_model = YOLO('yolov11s.pt')
                logger.info("YOLOv11 model downloaded and loaded successfully")
            except Exception as download_error:
                logger.error(
                    f"Failed to download YOLOv11 model: {download_error}")
                raise

    def _load_faster_rcnn_model(self):
        """Load the Faster R-CNN model"""
        try:
            from faster_rcnn_detector import FasterRCNNDetector
            self.faster_rcnn_detector = FasterRCNNDetector()
            logger.info("Faster R-CNN model loaded successfully")
        except ImportError as e:
            logger.error(f"Error importing Faster R-CNN detector: {e}")
            logger.warning("Faster R-CNN model not available")
            self.faster_rcnn_detector = None
        except Exception as e:
            logger.error(f"Error loading Faster R-CNN model: {e}")
            logger.warning("Faster R-CNN model not available")
            self.faster_rcnn_detector = None

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert image bytes to OpenCV format"""
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")
        return img

    def _generate_spectrum_data(self) -> List[float]:
        """Generate mock spectrum data (replace with real spectral analysis)"""
        data = []
        for i in range(100):
            # Simulate IR spectrum with peaks
            base_noise = random.random() * 0.1
            peak1 = np.exp(-((i - 25) / 8) ** 2) * 0.7
            peak2 = np.exp(-((i - 55) / 12) ** 2) * 0.9
            peak3 = np.exp(-((i - 80) / 6) ** 2) * 0.5
            data.append(float(base_noise + peak1 + peak2 + peak3))
        return data

    def _classify_particle_type(self, confidence: float, class_id: int) -> str:
        """Classify detected object into particle type"""
        # Simple heuristic based on confidence and class
        # In a real implementation, you'd use additional features
        if confidence > 0.8:
            return random.choice(['fiber', 'fragment'])
        elif confidence > 0.6:
            return random.choice(['film', 'pellet'])
        else:
            return 'foam'

    def _classify_polymer_type(self) -> str:
        """Classify polymer type (mock implementation)"""
        # In a real implementation, this would use spectral data
        return random.choice(self.config.POLYMER_TYPES)

    def detect_microplastics(self, image_bytes: bytes, mode: str = 'fast', algorithm: str = 'yolo') -> Dict[str, Any]:
        """
        Perform microplastic detection on image

        Args:
            image_bytes: Image data as bytes
            mode: Detection mode ('fast' or 'accurate')
            algorithm: Algorithm to use ('yolo' or 'faster_rcnn')

        Returns:
            Dictionary with detection results
        """
        try:
            if algorithm.lower() == 'faster_rcnn':
                if self.faster_rcnn_detector is None:
                    logger.warning(
                        "Faster R-CNN model not available, falling back to YOLO")
                    algorithm = 'yolo'
                else:
                    return self._detect_with_faster_rcnn(image_bytes, mode)

            # Default to YOLO detection
            return self._detect_with_yolo(image_bytes, mode)

        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise

    def _detect_with_yolo(self, image_bytes: bytes, mode: str = 'fast') -> Dict[str, Any]:
        """Perform detection using YOLOv11 model"""
        # Preprocess image
        img = self._preprocess_image(image_bytes)
        img_height, img_width = img.shape[:2]

        # Run YOLO detection
        conf_threshold = 0.3 if mode == 'fast' else 0.25
        results = self.yolo_model(img, conf=conf_threshold, iou=0.45)

        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    # Convert to application format
                    width = float(x2 - x1)
                    height = float(y2 - y1)

                    # Classify particle and polymer types
                    particle_type = self._classify_particle_type(
                        confidence, class_id)
                    polymer_type = self._classify_polymer_type()

                    detection = {
                        'id': f'det-{random.randint(1000, 9999)}',
                        'particleType': particle_type,
                        'polymerType': polymer_type,
                        'confidence': confidence,
                        'boundingBox': {
                            'x': float(x1),
                            'y': float(y1),
                            'width': width,
                            'height': height
                        },
                        'algorithm': 'yolo',
                        'ldirMatchScore': 0.75 + random.random() * 0.2,
                        'spectrumData': self._generate_spectrum_data()
                    }
                    detections.append(detection)

        # Calculate counts by type
        count_by_type = {ptype: 0 for ptype in self.config.PARTICLE_TYPES}
        for det in detections:
            if det['particleType'] in count_by_type:
                count_by_type[det['particleType']] += 1

        return {
            'detections': detections,
            'totalCount': len(detections),
            'countByType': count_by_type,
            'imageSize': {
                'width': int(img_width),
                'height': int(img_height)
            }
        }

    def _detect_with_faster_rcnn(self, image_bytes: bytes, mode: str = 'fast') -> Dict[str, Any]:
        """Perform detection using Faster R-CNN model"""
        # Use the Faster R-CNN detector
        results = self.faster_rcnn_detector.detect_microplastics(
            image_bytes, mode)

        # Add algorithm identifier to each detection
        for detection in results['detections']:
            detection['algorithm'] = 'faster_rcnn'

        return results


# Create singleton instance
detector = MicroplasticDetector()
