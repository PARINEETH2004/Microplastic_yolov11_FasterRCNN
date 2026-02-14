# ... existing imports ...
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
from detection import detector
from config import Config
import time
import uuid
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, origins=["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:5173", "http://127.0.0.1:5173"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# CRITICAL FIX: Add explicit configuration for file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


@app.route('/api/test-debug', methods=['POST'])
def test_debug():
    """Test endpoint to verify enhanced debugging works"""
    logger.info("=== TEST DEBUG ENDPOINT ===")
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"Files keys: {list(request.files.keys())}")
    logger.info(f"Form keys: {list(request.form.keys())}")
    logger.info(f"All files: {request.files}")
    logger.info(f"All form data: {request.form}")
    return jsonify({'message': 'Debug test successful', 'files': list(request.files.keys())})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})


@app.route('/api/detect', methods=['POST'])
def detect_microplastics():
    """Main detection endpoint"""
    logger.info("=== DETECT ENDPOINT CALLED ===")
    try:
        # ENHANCED DEBUGGING: Log detailed request information
        logger.info("=== ENHANCED REQUEST DEBUGGING ===")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Content-Length: {request.content_length}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Files keys: {list(request.files.keys())}")
        logger.info(f"Form keys: {list(request.form.keys())}")
        logger.info(f"All files: {request.files}")
        logger.info(f"All form data: {request.form}")

        # Check if request has the correct content type
        if not request.content_type or 'multipart/form-data' not in request.content_type:
            logger.error(f"Invalid content type: {request.content_type}")
            return jsonify({'error': f'Invalid content type: {request.content_type}. Expected multipart/form-data'}), 400

        # Validate request files
        if not request.files:
            logger.error("No files in request")
            return jsonify({'error': 'No files provided in request'}), 400

        # Validate request - CRITICAL FIX: Check multiple possible field names
        image_file = None
        possible_field_names = ['image', 'file', 'upload']

        for field_name in possible_field_names:
            if field_name in request.files:
                image_file = request.files[field_name]
                logger.info(f"Found image file in field '{field_name}'")
                break

        if not image_file:
            logger.error(
                f"No image file found. Available fields: {list(request.files.keys())}")
            return jsonify({'error': f'No image file provided. Available fields: {list(request.files.keys())}'}), 400

        # Log file details
        logger.info(f"File object type: {type(image_file)}")
        logger.info(f"File filename: {image_file.filename}")
        logger.info(f"File content type: {image_file.content_type}")

        # Check if file has content
        image_file.seek(0, 2)  # Seek to end
        file_size = image_file.tell()
        image_file.seek(0)  # Reset to beginning

        logger.info(f"File size: {file_size} bytes")

        if file_size == 0:
            logger.error("Uploaded file is empty")
            return jsonify({'error': 'Uploaded file is empty'}), 400

        # Validate filename
        if image_file.filename == '':
            logger.error("No image selected")
            return jsonify({'error': 'No image selected'}), 400

        # Validate file type
        if not allowed_file(image_file.filename):
            logger.error(f"File type not allowed: {image_file.filename}")
            return jsonify({'error': 'File type not allowed'}), 400

        # Get detection mode
        mode = request.form.get('mode', 'fast')
        if mode not in ['fast', 'accurate']:
            mode = 'fast'

        # Get detection algorithm
        algorithm = request.form.get('algorithm', 'yolo')
        if algorithm not in ['yolo', 'faster_rcnn']:
            algorithm = 'yolo'

        # Read image data
        image_bytes = image_file.read()
        if len(image_bytes) == 0:
            return jsonify({'error': 'Empty image file'}), 400

        # Generate image name
        filename = secure_filename(image_file.filename)
        image_name = f"{uuid.uuid4()}_{filename}"

        # Simulate processing time based on mode
        processing_time = 1.5 if mode == 'fast' else 3.0
        time.sleep(processing_time)

        # Perform detection
        logger.info(
            f"Processing image: {filename} with mode: {mode}, algorithm: {algorithm}")
        results = detector.detect_microplastics(image_bytes, mode, algorithm)

        # Return results
        response = {
            'imageUrl': f'/api/images/{image_name}',
            'imageName': filename,
            'timestamp': time.time(),
            'mode': mode,
            # Convert to milliseconds
            'processingTime': int(processing_time * 1000),
            'detections': results['detections'],
            'totalCount': results['totalCount'],
            'countByType': results['countByType'],
            'imageSize': results['imageSize']
        }

        logger.info(
            f"Detection completed: {results['totalCount']} detections found")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


@app.route('/api/images/<filename>', methods=['GET'])
def serve_image(filename):
    """Serve processed images (placeholder - in production you'd store images)"""
    # For now, return a placeholder response
    # In a real implementation, you'd serve actual processed images
    return jsonify({'message': 'Image serving not implemented in this demo'})


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get API configuration"""
    return jsonify({
        'apiVersion': app.config['API_VERSION'],
        'supportedModes': ['fast', 'accurate'],
        'maxFileSize': app.config['MAX_CONTENT_LENGTH'],
        'allowedExtensions': list(app.config['ALLOWED_EXTENSIONS'])
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File too large'}), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Microplastic Scout Backend...")
    logger.info(f"Model path: {app.config['MODEL_PATH']}")
    logger.info("Server starting on http://localhost:5000")

    # Disable debug mode to prevent auto-reload issues with PyTorch
    app.run(host='0.0.0.0', port=5000, debug=False)
