import os


class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get(
        'SECRET_KEY') or 'microplastic-scout-secret-key'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

    # Model configuration
    MODEL_PATH = os.path.join(os.path.dirname(
        __file__), 'models', 'yolov11_microplastic.pt')
    CONFIDENCE_THRESHOLD = 0.5

    # Class mappings (adjust based on your training)
    CLASS_NAMES = ['Microplastic']  # Update this based on your model's classes
    PARTICLE_TYPES = ['fiber', 'fragment', 'film', 'pellet', 'foam']
    POLYMER_TYPES = ['PE', 'PP', 'PS', 'PET', 'PVC', 'PA', 'Unknown']

    # Image processing
    IMAGE_SIZE = 640  # Input size for YOLO model

    # API configuration
    API_VERSION = 'v1'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
