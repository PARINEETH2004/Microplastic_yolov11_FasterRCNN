# ... existing imports ...
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
from detection import detector
from config import Config
import time
import uuid
from werkzeug.utils import secure_filename


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173", "http://127.0.0.1:5173"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])


app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': int(time.time())
    })


@app.route('/api/detect', methods=['POST'])
def detect_microplastics():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        mode = request.form.get('mode', 'fast')
        algorithm = request.form.get('algorithm', 'yolo')

        image_bytes = file.read()

        print(f"\n\n{'='*60}")
        print(f"DEBUG APP.PY: Received detection request")
        print(
            f"DEBUG APP.PY: Mode={mode}, Algorithm={algorithm}, Image size={len(image_bytes)} bytes")
        print(f"{'='*60}\n\n")

        start_time = time.time()
        result = detector.detect_microplastics(image_bytes, mode, algorithm)
        processing_time = int((time.time() - start_time) * 1000)

        print(f"\n\n{'='*60}")
        print(f"DEBUG APP.PY: Detection completed")
        print(
            f"DEBUG APP.PY: Total count returned: {result.get('totalCount', 0)}")
        print(f"{'='*60}\n\n")

        result['processingTime'] = processing_time
        result['mode'] = mode
        result['algorithm'] = algorithm

        return jsonify(result)

    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': 'Detection failed'}), 500


@app.route('/api/images/<filename>')
def serve_processed_image(filename):
    try:
        return send_from_directory(app.config['PROCESSED_IMAGES_FOLDER'], filename)
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return jsonify({'error': 'Image not found'}), 404


if __name__ == '__main__':
    logger.info("Starting Microplastic Scout Backend...")
    logger.info(f"Model path: {app.config['MODEL_PATH']}")
    logger.info("Server starting on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
