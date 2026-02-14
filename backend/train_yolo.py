import os
import sys
from ultralytics import YOLO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_microplastic_detector():
    """Train YOLOv11 on microplastic dataset"""

    # Define paths
    dataset_dir = r"C:\Users\PARINEETH\Downloads\microplastic-scout-main\microplastic images"
    data_yaml = os.path.join(dataset_dir, "data.yaml")

    # Verify dataset exists
    if not os.path.exists(data_yaml):
        logger.error(f"Dataset not found at {data_yaml}")
        return False

    logger.info(f"Using dataset: {data_yaml}")

    try:
        # Load a pretrained YOLOv11 model
        model = YOLO('yolov11s.pt')  # or yolov11n.pt for smaller model

        logger.info("Starting training...")

        # Train the model
        results = model.train(
            data=data_yaml,           # dataset YAML file
            epochs=100,               # number of training epochs
            imgsz=640,                # input image size
            batch=16,                 # batch size
            name='microplastic_v11',  # experiment name
            project='runs/detect',    # project directory
            patience=20,              # early stopping patience
            save_period=10,           # save checkpoint every 10 epochs
            device='cpu',             # use CPU (change to 0 for GPU)
            verbose=True              # verbose output
        )

        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {results.save_dir}/weights/best.pt")

        # Validate the model
        logger.info("Validating model...")
        metrics = model.val()
        logger.info(f"Validation mAP50: {metrics.box.map50}")
        logger.info(f"Validation mAP50-95: {metrics.box.map}")

        return True

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return False


def export_model():
    """Export trained model to different formats"""
    try:
        # Load the trained model
        model = YOLO('runs/detect/microplastic_v11/weights/best.pt')

        # Export to ONNX (for web deployment)
        logger.info("Exporting to ONNX...")
        model.export(format='onnx')

        # Export to TorchScript
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

    # Train the model
    success = train_microplastic_detector()

    if success:
        print("\nTraining completed! Now exporting model...")
        export_model()
        print("\n🎉 All done! Your trained model is ready for use.")
    else:
        print("\n❌ Training failed. Check the logs above for details.")
