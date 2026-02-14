import requests
import os


def download_yolo_model():
    """Download YOLOv11s model if not present"""
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
    model_path = "yolov11s.pt"

    if os.path.exists(model_path):
        print("Model already exists")
        return True

    print("Downloading YOLOv11s model...")
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print("Model downloaded successfully")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


if __name__ == "__main__":
    download_yolo_model()
