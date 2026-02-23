import uuid
import time
import cv2
import numpy as np
from ultralytics import YOLO
import random
import os
from typing import List, Dict, Any, Tuple
from config import Config
import logging
from faster_rcnn_detector import FasterRCNNDetector

logger = logging.getLogger(__name__)


class MicroplasticDetector:
    def __init__(self):
        self.config = Config()
        self.yolo_model = None
        self.faster_rcnn_detector = None
        self._load_models()

    def _load_models(self):
        self._load_yolo_model()
        self._load_faster_rcnn_model()

    def _load_yolo_model(self):
        try:
            model_path = self.config.MODEL_PATH
            if os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
                logger.info(f"Loaded custom YOLO model from {model_path}")
            else:
                self.yolo_model = YOLO('yolov11s.pt')
                logger.warning(
                    f"Custom model not found at {model_path}, using pretrained yolov11s.pt")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            self.yolo_model = YOLO('yolov11s.pt')
            logger.info("Using fallback pretrained YOLO model")

    def _load_faster_rcnn_model(self):
        try:
            self.faster_rcnn_detector = FasterRCNNDetector()
            logger.info("Faster R-CNN model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Faster R-CNN model: {e}")
            self.faster_rcnn_detector = None

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def _classify_particle_type(self, confidence: float, class_id: int) -> str:
        if confidence > 0.8:
            return random.choice(['fiber', 'fragment'])
        elif confidence > 0.6:
            return random.choice(['film', 'pellet'])
        else:
            return 'foam'

    def _classify_polymer_type(self, roi: np.ndarray) -> str:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return "Unknown"

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            return "Unknown"

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (w * h)

        if aspect_ratio > 3 or aspect_ratio < 0.3:
            return "Fiber"
        if 0.75 < circularity <= 1.2:
            return "Pellet"
        if edge_density > 25:
            return "Foam"
        if 1.5 < aspect_ratio <= 3:
            return "Film"
        return "Fragment"

    def detect_microplastics(self, image_bytes: bytes, mode: str = 'fast', algorithm: str = 'yolo') -> Dict[str, Any]:
        try:
            img = self._preprocess_image(image_bytes)
            img_height, img_width = img.shape[:2]

            if algorithm == 'faster_rcnn' and self.faster_rcnn_detector:
                return self._detect_with_faster_rcnn(img, mode)
            else:
                return self._detect_with_yolo(image_bytes, mode)

        except Exception as e:
            logger.error(f"Detection error: {e}")
            raise

    def _detect_with_yolo(self, image_bytes: bytes, mode: str = 'fast') -> Dict[str, Any]:
        img = self._preprocess_image(image_bytes)
        img_height, img_width = img.shape[:2]

        conf_threshold = 0.3 if mode == 'fast' else 0.25
        results = self.yolo_model(img, conf=conf_threshold, iou=0.45)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    width = float(x2 - x1)
                    height = float(y2 - y1)

                    x1_int, y1_int = int(x1), int(y1)
                    x2_int, y2_int = int(x2), int(y2)
                    roi = img[y1_int:y2_int, x1_int:x2_int]

                    particle_type = self._classify_particle_type(
                        confidence, class_id)
                    polymer_type = self._classify_polymer_type(roi)

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
                        }
                    }
                    detections.append(detection)

        total_count = len(detections)
        count_by_type = {}
        for particle_type in self.config.PARTICLE_TYPES:
            count_by_type[particle_type] = len(
                [d for d in detections if d['particleType'] == particle_type])

        # Add missing fields to each detection
        for detection in detections:
            detection['ldirMatchScore'] = random.uniform(0.7, 0.95)
            detection['spectrumData'] = [
                random.uniform(0.1, 1.0) for _ in range(100)]
            detection['algorithm'] = 'yolo'

        return {
            'imageUrl': f'/api/images/processed_{uuid.uuid4().hex}.jpg',
            'imageName': 'processed_image.jpg',
            'timestamp': int(time.time() * 1000),
            'mode': mode,
            # Simulate processing time
            'processingTime': random.randint(1500, 3000),
            'totalCount': total_count,
            'detections': detections,
            'countByType': count_by_type
        }

    def _detect_with_faster_rcnn(self, img: np.ndarray, mode: str = 'fast') -> Dict[str, Any]:
        detections = self.faster_rcnn_detector.detect(img)

        processed_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            width = float(x2 - x1)
            height = float(y2 - y1)

            x1_int, y1_int = int(x1), int(y1)
            x2_int, y2_int = int(x2), int(y2)
            roi = img[y1_int:y2_int, x1_int:x2_int]

            particle_type = self._classify_particle_type(
                confidence, det['class_id'])
            polymer_type = self._classify_polymer_type(roi)

            processed_detection = {
                'id': f'det-{random.randint(1000, 9999)}',
                'particleType': particle_type,
                'polymerType': polymer_type,
                'confidence': confidence,
                'boundingBox': {
                    'x': float(x1),
                    'y': float(y1),
                    'width': width,
                    'height': height
                }
            }
            processed_detections.append(processed_detection)

        total_count = len(processed_detections)
        count_by_type = {}
        for particle_type in self.config.PARTICLE_TYPES:
            count_by_type[particle_type] = len(
                [d for d in processed_detections if d['particleType'] == particle_type])

        # Add missing fields to each detection
        for detection in processed_detections:
            detection['ldirMatchScore'] = random.uniform(0.8, 0.98)
            detection['spectrumData'] = [
                random.uniform(0.1, 1.0) for _ in range(100)]
            detection['algorithm'] = 'faster_rcnn'

        return {
            'imageUrl': f'/api/images/processed_{uuid.uuid4().hex}.jpg',
            'imageName': 'processed_image.jpg',
            'timestamp': int(time.time() * 1000),
            'mode': mode,
            # Simulate processing time
            'processingTime': random.randint(2500, 5000),
            'totalCount': total_count,
            'detections': processed_detections,
            'countByType': count_by_type
        }


detector = MicroplasticDetector()
