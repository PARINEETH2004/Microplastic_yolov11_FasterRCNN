import os
import sys
from ultralytics import YOLO
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_microplastic_detector():
    dataset_dir = os.path.join(os.path.dirname(
        __file__), '..', 'microplastic images')
    data_yaml = os.path.join(dataset_dir, "data.yaml")

    if not os.path.exists(data_yaml):
        logger.error(f"Dataset not found at {data_yaml}")
        return False

    logger.info(f"Using dataset: {data_yaml}")

    try:
        model = YOLO('yolov11s.pt')

        logger.info("Starting training...")

        results = model.train(
            data=data_yaml,
            epochs=100,
            imgsz=640,
            batch=16,
            name='microplastic_v11',
            project='runs/detect',
            patience=20,
            save_period=10,
            device='cpu',
            verbose=True
        )

        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {results.save_dir}/weights/best.pt")

        logger.info("Validating model...")
        metrics = model.val()
        logger.info(f"Validation mAP50: {metrics.box.map50}")
        logger.info(f"Validation mAP50-95: {metrics.box.map}")

        return True

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False


def export_model():
    try:
        model = YOLO('runs/detect/microplastic_v11/weights/best.pt')

        logger.info("Exporting to ONNX...")
        model.export(format='onnx')

        logger.info("Exporting to TorchScript...")
        model.export(format='torchscript')

        logger.info("Model export completed!")
        return True

    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Microplastic YOLOv11 Training Script")
    print("=" * 40)

    success = train_microplastic_detector()

    if success:
        print("\nTraining completed! Now exporting model...")
        export_model()
        print("\n🎉 All done! Your trained model is ready for use.")
    else:
        print("\n❌ Training failed. Check the logs above for details.")
