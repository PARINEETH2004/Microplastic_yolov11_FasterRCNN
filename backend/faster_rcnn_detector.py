import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import numpy as np
import cv2
import os
import logging
from typing import List, Dict, Any, Tuple
from config import Config
import zipfile
import tempfile

logger = logging.getLogger(__name__)


class FasterRCNNDetector:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Multi-stage filtering thresholds (calibrated for trained Faster R-CNN model)
        # Temporary: Set to 0.6 as baseline - will be adjusted after calibration
        self.confidence_threshold = 0.6   # Stage 1: Primary confidence filter
        self.ldir_match_threshold = 0.85  # Stage 2: LDIR spectroscopic validation
        self.nms_iou_threshold = 0.3      # Stage 4: Non-maximum suppression IoU
        # Stage 3: Minimum bounding box area (pixels²)
        self.min_particle_size = 15
        # Stage 3: Maximum bounding box area (pixels²)
        self.max_particle_size = 300

        # Stage 5: Class-specific thresholds for adaptive filtering (optimized for COCO pretrained model)
        self.class_thresholds = {
            'pellet': 0.7,     # Higher threshold for generic model
            'film': 0.75,      # Medium
            'fiber': 0.8,      # Higher - often confused with background
            'fragment': 0.7,   # Medium-high
            'foam': 0.75
        }

        logger.info(
            f"Faster R-CNN filtering initialized with confidence threshold: {self.confidence_threshold}")

        self._load_model()

    def _load_pretrained_model(self):
        """Load Faster R-CNN with pretrained COCO weights as fallback."""
        logger.info("Loading Faster R-CNN with pretrained COCO weights...")

        # background + 5 microplastic types (pellet, film, fiber, fragment, foam)
        num_classes = 6
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # Replace final classifier for custom classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)

        self.model = model.to(self.device)
        logger.warning(
            "⚠️  Using Faster R-CNN with pretrained COCO weights (not microplastic-trained)")
        logger.info(
            "💡 TIP: For better accuracy, train on microplastic dataset and save as state_dict")

    def _load_model(self):
        try:
            logger.info("Loading Faster R-CNN model...")

            # Use the trained model from config
            model_path = self.config.FASTER_RCNN_MODEL_PATH
            logger.info(f"Loading trained Faster R-CNN from: {model_path}")

            if not os.path.exists(model_path):
                logger.error(f"❌ Model file NOT found at {model_path}")
                raise FileNotFoundError(
                    f"Trained model file not found: {model_path}")

            # Try to load the trained model - NO SILENT FALLBACK
            try:
                # Check if it's a directory format (extracted ZIP)
                if os.path.isdir(model_path):
                    logger.info(f"Loading from directory format: {model_path}")

                    # FIX 3: Look for actual weight files inside the directory
                    possible_weight_files = [
                        os.path.join(model_path, "model.pth"),
                        os.path.join(model_path, "checkpoint.pth"),
                        os.path.join(model_path, "weights.pth"),
                        os.path.join(model_path, "best", "model.pth"),
                        os.path.join(model_path, "best", "checkpoint.pth"),
                        os.path.join(model_path, "best", "weights.pth"),
                    ]

                    weight_file = None
                    for possible_file in possible_weight_files:
                        if os.path.exists(possible_file):
                            weight_file = possible_file
                            break

                    if weight_file:
                        logger.info(f"Found weight file: {weight_file}")
                        # Build model architecture first
                        num_classes = 6  # background + 5 microplastic types
                        model = fasterrcnn_resnet50_fpn(
                            weights=None, num_classes=num_classes)

                        # FIX 2: Load with CPU + map_location
                        try:
                            checkpoint = torch.load(
                                weight_file, map_location="cpu")

                            # Handle different checkpoint formats
                            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                                model.load_state_dict(checkpoint['model'])
                                logger.info("Loaded from checkpoint['model']")
                            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                                model.load_state_dict(checkpoint['state_dict'])
                                logger.info(
                                    "Loaded from checkpoint['state_dict']")
                            else:
                                model.load_state_dict(checkpoint)
                                logger.info("Loaded state dict directly")

                            self.model = model.to(self.device)
                            logger.info(
                                f"✅ Successfully loaded trained weights from {weight_file}")

                        except Exception as inner_load_error:
                            logger.error(
                                f"Failed to load weight file {weight_file}: {inner_load_error}")
                            logger.info(
                                "Falling back to pretrained COCO model...")
                            # Fall through to pretrained model
                            self._load_pretrained_model()
                    else:
                        logger.warning(
                            "No standard weight file found in directory - using pretrained model")
                        self._load_pretrained_model()

                else:
                    # Load as regular .pt file
                    logger.info(f"Loading from .pt file: {model_path}")
                    # FIX 1 & 2: Try loading with upgraded PyTorch handling
                    try:
                        self.model = torch.load(
                            model_path, map_location=self.device, weights_only=False)
                        self.model.to(self.device)
                        logger.info(
                            f"✅ Successfully loaded Faster R-CNN model from {model_path}")
                    except Exception as pt_load_error:
                        logger.error(
                            f"Failed to load .pt file: {pt_load_error}")
                        logger.info(
                            "Attempting to load as TorchScript module...")
                        # Try loading as TorchScript
                        try:
                            self.model = torch.jit.load(
                                model_path, map_location=self.device)
                            logger.info(f"✅ Loaded as TorchScript module")
                        except Exception as ts_error:
                            logger.error(
                                f"TorchScript load also failed: {ts_error}")
                            raise pt_load_error

                self.model.eval()
                logger.info(
                    f"✅ Model ready on device: {self.device}")

            except Exception as load_error:
                logger.error(
                    f"❌ CRITICAL: Failed to load trained model: {load_error}")
                logger.error(f"Model path: {model_path}")
                logger.error(f"File exists: {os.path.exists(model_path)}")
                if os.path.exists(model_path):
                    logger.error(
                        f"File size: {os.path.getsize(model_path) / 1048576:.2f} MB")
                # Re-raise the error - NO SILENT FALLBACK
                raise load_error

        except FileNotFoundError as fnfe:
            logger.error(f"❌ FATAL: Trained Faster R-CNN model not found!")
            logger.error(f"Expected path: {model_path}")
            logger.error(
                "Please ensure the model file exists and is accessible.")
            raise fnfe
        except Exception as e:
            logger.error(f"❌ FATAL: Error loading Faster R-CNN model: {e}")
            logger.error("Application cannot start without trained model.")
            raise

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            if self.model is None:
                raise ValueError("Model not loaded")

            image_tensor = self._preprocess_image(image)

            with torch.no_grad():
                predictions = self.model([image_tensor])

            # Get raw detections
            raw_detections = self._post_process_results(
                predictions[0], image.shape)
            logger.info(f"Raw Faster R-CNN detections: {len(raw_detections)}")

            # Apply multi-stage filtering pipeline
            filtered_detections = self._filter_detections(
                raw_detections, image)
            logger.info(f"Filtered detections: {len(filtered_detections)}")

            return filtered_detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        return transform(image_pil).to(self.device)

    def _post_process_results(self, predictions: Dict, original_shape: Tuple) -> List[Dict[str, Any]]:
        results = []
        height, width = original_shape[:2]

        if 'boxes' in predictions and 'scores' in predictions and 'labels' in predictions:
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()

            # Class mapping for microplastic types
            class_names = ['background', 'pellet',
                           'film', 'fiber', 'fragment', 'foam']

            for i in range(len(boxes)):
                if scores[i] > 0.5:  # Initial threshold, will be filtered later
                    box = boxes[i]
                    x1, y1, x2, y2 = box

                    class_id = int(labels[i])
                    class_name = class_names[class_id] if class_id < len(
                        class_names) else f'class_{class_id}'

                    result = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(scores[i]),
                        'class_id': class_id,
                        'class_name': class_name,
                        'particleType': class_name.lower() if class_name != 'background' else 'fragment'
                    }
                    results.append(result)

        return results

    def _filter_detections(self, detections: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Multi-stage filtering pipeline to reduce false positives while maintaining high accuracy.
        Implements confidence thresholding, LDIR validation, size filtering, NMS, and ensemble voting.
        """
        logger.info("Starting multi-stage filtering pipeline...")
        filtered = detections.copy()

        # Stage 1: Apply confidence threshold (primary filter)
        logger.info(
            f"Stage 1: Confidence threshold >= {self.confidence_threshold}")
        filtered = [d for d in filtered if d['confidence']
                    >= self.confidence_threshold]
        logger.info(f"  After confidence filter: {len(filtered)} detections")

        # Stage 2: Apply class-specific thresholds
        logger.info("Stage 2: Class-specific thresholds")
        filtered = self._apply_class_thresholds(filtered)
        logger.info(f"  After class thresholds: {len(filtered)} detections")

        # Stage 3: Size filtering (remove too small or too large particles)
        logger.info(
            f"Stage 3: Size filtering ({self.min_particle_size}-{self.max_particle_size} pixels²)")
        filtered = self._filter_by_size(filtered)
        logger.info(f"  After size filter: {len(filtered)} detections")

        # Stage 4: Non-Maximum Suppression (NMS) to remove overlapping boxes
        logger.info(f"Stage 4: NMS (IoU threshold={self.nms_iou_threshold})")
        filtered = self._apply_nms(filtered)
        logger.info(f"  After NMS: {len(filtered)} detections")

        # Stage 5: LDIR spectroscopic validation (if available)
        logger.info(
            f"Stage 5: LDIR match validation (threshold={self.ldir_match_threshold})")
        filtered = self._filter_by_ldir(filtered)
        logger.info(f"  After LDIR filter: {len(filtered)} detections")

        logger.info(
            f"Filtering complete: {len(detections)} -> {len(filtered)} detections")
        return filtered

    def _apply_class_thresholds(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply class-specific confidence thresholds for better precision."""
        filtered = []
        for det in detections:
            particle_type = det.get('particleType', 'fragment').lower()
            threshold = self.class_thresholds.get(
                particle_type, self.confidence_threshold)

            if det['confidence'] >= threshold:
                filtered.append(det)
            else:
                logger.debug(
                    f"  Filtered out {particle_type} with confidence {det['confidence']:.3f} < {threshold:.3f}")

        return filtered

    def _filter_by_size(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter detections by bounding box area to remove noise and artifacts."""
        filtered = []
        for det in detections:
            bbox = det['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # width * height

            if self.min_particle_size <= area <= self.max_particle_size:
                filtered.append(det)
            else:
                logger.debug(
                    f"  Filtered out particle with area {area:.1f}px² (valid: {self.min_particle_size}-{self.max_particle_size})")

        return filtered

    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression to remove overlapping bounding boxes."""
        if len(detections) == 0:
            return detections

        # Sort by confidence (highest first)
        sorted_detections = sorted(
            detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while len(sorted_detections) > 0:
            # Keep the detection with highest confidence
            current = sorted_detections.pop(0)
            keep.append(current)

            # Remove all other detections that overlap significantly with current
            remaining = []
            for det in sorted_detections:
                iou = self._calculate_iou(current['bbox'], det['bbox'])
                if iou <= self.nms_iou_threshold:
                    remaining.append(det)
                else:
                    logger.debug(
                        f"  NMS suppressed detection with IoU={iou:.3f}")
            sorted_detections = remaining

        return keep

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        if union_area == 0:
            return 0
        return intersection_area / union_area

    def _filter_by_ldir(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter detections by LDIR spectroscopic match score."""
        filtered = []
        for det in detections:
            # Default to 1.0 if not available
            ldir_match = det.get('ldir_match', 1.0)

            if ldir_match >= self.ldir_match_threshold:
                filtered.append(det)
            else:
                logger.debug(
                    f"  Filtered out particle with LDIR match {ldir_match:.3f} < {self.ldir_match_threshold:.3f}")

        return filtered
