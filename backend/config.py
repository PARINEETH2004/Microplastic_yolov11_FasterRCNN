import os


class Config:
    SECRET_KEY = os.environ.get(
        'SECRET_KEY') or 'microplastic-scout-secret-key'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    MODEL_PATH = os.path.join(os.path.dirname(
        __file__), 'models', 'yolov11_microplastic.pt')
    CONFIDENCE_THRESHOLD = 0.5

    CLASS_NAMES = ['Microplastic']
    PARTICLE_TYPES = ['fiber', 'fragment', 'film', 'pellet', 'foam']
    POLYMER_TYPES = ['PE', 'PP', 'PS', 'PET', 'PVC', 'PA', 'Unknown']

    IMAGE_SIZE = 640

    API_VERSION = 'v1'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
