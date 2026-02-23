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
        self._load_model()

    def _load_model(self):
        try:
            logger.info("Loading Faster R-CNN model...")

            # Define model architecture
            num_classes = 2
            model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes)

            # Try multiple loading approaches
            success = False

            # Approach 1: Try loading from ZIP file
            zip_path = os.path.join(os.path.dirname(
                __file__), 'models', 'fasterRCNN_best.pt.zip')
            if os.path.exists(zip_path):
                try:
                    logger.info(f"Attempting to load from ZIP: {zip_path}")
                    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            # Extract the main model file
                            model_files = [f for f in zip_ref.namelist(
                            ) if f.endswith('.pt') or 'model' in f]
                            if model_files:
                                zip_ref.extract(model_files[0], tmp_file.name)
                                tmp_file.flush()

                                # Try to load as TorchScript or regular model
                                try:
                                    loaded_model = torch.jit.load(
                                        tmp_file.name, map_location=self.device)
                                    self.model = loaded_model
                                    logger.info(
                                        "✅ Successfully loaded Faster R-CNN from ZIP (TorchScript)")
                                    success = True
                                except:
                                    # Try regular torch.load
                                    checkpoint = torch.load(
                                        tmp_file.name, map_location=self.device)
                                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                                        model.load_state_dict(
                                            checkpoint['model_state_dict'])
                                        logger.info(
                                            "✅ Successfully loaded Faster R-CNN from ZIP (state dict)")
                                        success = True
                                    elif hasattr(checkpoint, 'state_dict'):
                                        model.load_state_dict(
                                            checkpoint.state_dict())
                                        logger.info(
                                            "✅ Successfully loaded Faster R-CNN from ZIP (model object)")
                                        success = True
                        os.unlink(tmp_file.name)
                except Exception as e:
                    logger.warning(f"ZIP loading failed: {e}")

            # Approach 2: Try loading from directory format
            if not success:
                model_dir = os.path.join(os.path.dirname(
                    __file__), 'models', 'fasterRCNN_best.pt', 'best')
                data_pkl_path = os.path.join(model_dir, 'data.pkl')

                if os.path.exists(data_pkl_path):
                    try:
                        logger.info(
                            "Attempting to load from directory format...")
                        # Try to load with minimal approach
                        import pickle

                        # Simple approach - just try to load and see what we get
                        with open(data_pkl_path, 'rb') as f:
                            # Try to load without custom unpickler first
                            try:
                                checkpoint = pickle.load(f)
                                if isinstance(checkpoint, dict):
                                    state_dict = checkpoint.get(
                                        'model_state_dict') or checkpoint.get('state_dict')
                                    if state_dict:
                                        model.load_state_dict(
                                            state_dict, strict=False)
                                        logger.info(
                                            "✅ Successfully loaded Faster R-CNN from directory (state dict)")
                                        success = True
                            except:
                                # If that fails, try with minimal storage handling
                                class SimpleStorageUnpickler(pickle.Unpickler):
                                    def persistent_load(self, saved_id):
                                        return None  # Return None for storage objects

                                f.seek(0)  # Reset file pointer
                                unpickler = SimpleStorageUnpickler(f)
                                checkpoint = unpickler.load()

                                if isinstance(checkpoint, dict):
                                    state_dict = checkpoint.get(
                                        'model_state_dict') or checkpoint.get('state_dict')
                                    if state_dict:
                                        # Filter out problematic keys
                                        filtered_state_dict = {k: v for k, v in state_dict.items()
                                                               if not (hasattr(v, 'dtype') and str(v.dtype) == 'object')}
                                        try:
                                            model.load_state_dict(
                                                filtered_state_dict, strict=False)
                                            logger.info(
                                                "✅ Successfully loaded Faster R-CNN from directory (filtered)")
                                            success = True
                                        except Exception as filter_error:
                                            logger.warning(
                                                f"Filtered loading failed: {filter_error}")
                        f.close()
                    except Exception as dir_error:
                        logger.warning(
                            f"Directory loading failed: {dir_error}")

            # Finalize model
            if success:
                model.to(self.device)
                model.eval()
                self.model = model
                logger.info("Faster R-CNN model loaded with trained weights ✅")
            else:
                logger.warning(
                    "Could not load trained weights, using pretrained model")
                model.to(self.device)
                model.eval()
                self.model = model

        except Exception as e:
            logger.error(f"Error loading Faster R-CNN: {e}")
            # Emergency fallback
            num_classes = 2
            model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes)
            model.to(self.device)
            model.eval()
            self.model = model
            logger.warning("Emergency fallback model created")

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            if self.model is None:
                raise ValueError("Model not loaded")

            image_tensor = self._preprocess_image(image)

            with torch.no_grad():
                predictions = self.model([image_tensor])

            results = self._post_process_results(predictions[0], image.shape)
            return results

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

            for i in range(len(boxes)):
                if scores[i] > 0.5:
                    box = boxes[i]
                    x1, y1, x2, y2 = box

                    result = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(scores[i]),
                        'class_id': int(labels[i]),
                        'class_name': 'microplastic' if labels[i] == 1 else 'background'
                    }
                    results.append(result)

        return results
