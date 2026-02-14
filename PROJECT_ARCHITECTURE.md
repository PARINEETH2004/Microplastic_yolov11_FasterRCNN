# Microplastic Scout - Project Architecture & Implementation Guide

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Component Connections](#component-connections)
- [YOLOv11 Implementation](#yolov11-implementation)
- [Data Flow](#data-flow)
- [Technical Stack](#technical-stack)
- [Development Setup](#development-setup)

## Project Overview

Microplastic Scout is a full-stack application designed for automated microplastic particle detection and classification in environmental samples. The system combines computer vision (YOLOv11) with a modern web interface to provide researchers and environmental scientists with an accessible tool for microplastic analysis.

## System Architecture

```
┌─────────────────┐    HTTP/REST    ┌──────────────────┐
│   React Frontend│ ←──────────────→ │  Flask Backend   │
│   (Port 8081)   │                 │   (Port 5000)    │
└─────────────────┘                 └──────────────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │  YOLOv11 Model   │
                                  │  (Detection)     │
                                  └──────────────────┘
                                            │
                                  ┌──────────────────┐
                                  │   Image Processing│
                                  │   (OpenCV)       │
                                  └──────────────────┘
```

### Key Components:

1. **Frontend (React/TypeScript)**
   - User interface for image upload and result visualization
   - Real-time detection overlay with bounding boxes
   - Interactive results display with particle classification

2. **Backend (Python Flask)**
   - RESTful API for image processing requests
   - YOLOv11 model integration and inference
   - Image preprocessing and result formatting

3. **Computer Vision Engine**
   - YOLOv11 object detection model
   - Custom-trained weights for microplastic detection
   - Real-time inference pipeline

## Component Connections

### 1. Frontend → Backend Communication

**API Endpoints:**
```
POST /api/detect     → Image analysis request
GET  /api/health     → System health check
GET  /api/config     → API configuration
```

**Request Flow:**
```typescript
// Frontend (src/lib/yoloDetection.ts)
const formData = new FormData();
formData.append('image', imageFile);
formData.append('mode', detectionMode);

const response = await fetch('/api/detect', {
  method: 'POST',
  body: formData
});

// Backend (backend/app.py)
@app.route('/api/detect', methods=['POST'])
def detect_microplastics():
    image_file = request.files['image']
    mode = request.form.get('mode', 'fast')
    results = detector.detect_microplastics(image_file.read(), mode)
    return jsonify(results)
```

### 2. Backend → YOLO Model Integration

**Model Loading Pipeline:**
```python
# backend/detection.py
class MicroplasticDetector:
    def __init__(self):
        self._load_model()
    
    def _load_model(self):
        # Priority loading sequence:
        # 1. Custom trained model (best.pt)
        trained_model = 'runs/detect/microplastic_v11/weights/best.pt'
        # 2. Custom model file
        custom_model = 'models/yolov11_microplastic.pt'
        # 3. Pretrained YOLOv11s fallback
        pretrained_model = 'yolov11s.pt'
```

### 3. Data Processing Pipeline

**Image Flow:**
```
Uploaded Image → Preprocessing → YOLO Inference → Post-processing → Results
     (File)         (OpenCV)      (Model)         (Formatting)    (JSON)
```

## YOLOv11 Implementation

### Model Architecture

YOLOv11 (You Only Look Once version 11) is the latest iteration of the real-time object detection system. Key features:

- **Single-stage detector**: Processes entire image in one forward pass
- **Real-time performance**: Optimized for speed and accuracy
- **Multi-scale detection**: Detects objects at various sizes
- **Anchor-free design**: Simplified architecture for better generalization

### Model Files Structure

```
backend/
├── models/
│   └── yolov11_microplastic.pt      # Custom trained model (if available)
├── runs/
│   └── detect/
│       └── microplastic_v11/
│           └── weights/
│               ├── best.pt          # Best performing weights
│               ├── last.pt          # Last epoch weights
│               └── epoch*.pt        # Intermediate checkpoints
├── yolov11s.pt                      # Pretrained base model
└── detection.py                     # Model integration logic
```

### Training Data Structure

```
microplastic images/
├── train/
│   ├── images/     # Training images
│   └── labels/     # YOLO format annotations (.txt files)
├── valid/
│   ├── images/     # Validation images
│   └── labels/     # Validation annotations
└── test/
    ├── images/     # Test images
    └── labels/     # Test annotations
```

### YOLO Configuration

**data.yaml (Dataset Configuration):**
```yaml
path: ../microplastic images
train: train/images
val: valid/images
test: test/images

nc: 1  # Number of classes
names: ['Microplastic']  # Class names
```

**Model Configuration (config.py):**
```python
class Config:
    MODEL_PATH = 'models/yolov11_microplastic.pt'
    CONFIDENCE_THRESHOLD = 0.5
    IMAGE_SIZE = 640  # Input size for YOLO model
    CLASS_NAMES = ['Microplastic']
    PARTICLE_TYPES = ['fiber', 'fragment', 'film', 'pellet', 'foam']
    POLYMER_TYPES = ['PE', 'PP', 'PS', 'PET', 'PVC', 'PA', 'Unknown']
```

### Detection Process

**Inference Pipeline:**
```python
def detect_microplastics(self, image_bytes: bytes, mode: str) -> Dict:
    # 1. Preprocess image
    img = self._preprocess_image(image_bytes)  # OpenCV conversion
    
    # 2. Run YOLO detection
    conf_threshold = 0.3 if mode == 'fast' else 0.25
    results = self.model(img, conf=conf_threshold, iou=0.45)
    
    # 3. Process results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            
            # Classify particle type
            particle_type = self._classify_particle_type(confidence, class_id)
            polymer_type = self._classify_polymer_type()
            
            detections.append({
                'boundingBox': {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1},
                'confidence': confidence,
                'particleType': particle_type,
                'polymerType': polymer_type
            })
```

### Model Training Process

**Training Command:**
```bash
# Using the train_yolo.py script
python train_yolo.py --data data.yaml --epochs 100 --img 640
```

**Training Configuration:**
- **Epochs**: 100 (configurable)
- **Image size**: 640×640 pixels
- **Batch size**: Auto-optimized
- **Augmentation**: Standard YOLO augmentations
- **Validation**: Performed every epoch

## Data Flow

### 1. Image Upload Flow

```
User Uploads Image
       ↓
Frontend (File API)
       ↓
FormData Creation
       ↓
HTTP POST to /api/detect
       ↓
Backend Validation
       ↓
Image Preprocessing (OpenCV)
       ↓
YOLO Inference
       ↓
Result Processing
       ↓
JSON Response
       ↓
Frontend Visualization
```

### 2. Detection Result Structure

```json
{
  "imageUrl": "/api/images/uuid_filename.jpg",
  "imageName": "sample_image.jpg",
  "timestamp": 1234567890,
  "mode": "fast",
  "processingTime": 1500,
  "totalCount": 23,
  "detections": [
    {
      "id": "det-1234",
      "particleType": "fiber",
      "polymerType": "PE",
      "confidence": 0.87,
      "boundingBox": {
        "x": 120,
        "y": 85,
        "width": 45,
        "height": 120
      },
      "ldirMatchScore": 0.92,
      "spectrumData": [0.1, 0.3, 0.7, ...]
    }
  ],
  "countByType": {
    "fiber": 12,
    "fragment": 8,
    "film": 2,
    "pellet": 1,
    "foam": 0
  }
}
```

### 3. Visualization Pipeline

```
Detection Results
       ↓
DetectionOverlay Component
       ↓
SVG Overlay Generation
       ↓
Bounding Box Rendering
       ↓
Label Placement
       ↓
Interactive Selection
```

## Technical Stack

### Frontend
- **React 18** - Component-based UI framework
- **TypeScript** - Type safety and better developer experience
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - Accessible UI components
- **React Query** - Server state management

### Backend
- **Python 3.8+** - Core programming language
- **Flask** - Lightweight web framework
- **Flask-CORS** - Cross-origin resource sharing
- **Ultralytics YOLO** - Computer vision framework
- **OpenCV** - Image processing library
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing

### Computer Vision
- **YOLOv11** - Object detection model
- **Pretrained weights**: yolov11s.pt
- **Custom training**: microplastic_v11 dataset
- **Inference engine**: Ultralytics framework

## Development Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup
```bash
cd microplastic-scout-main
npm install
npm run dev
```

### Environment Variables
```bash
# Backend (.env)
FLASK_ENV=development
SECRET_KEY=your-secret-key

# Frontend (.env)
VITE_API_BASE_URL=http://localhost:5000
```

## Model Performance

### Detection Modes
- **Fast Mode**: 
  - Confidence threshold: 0.3
  - Processing time: ~1.5 seconds
  - Lower precision, higher recall

- **Accurate Mode**: 
  - Confidence threshold: 0.25
  - Processing time: ~3.0 seconds
  - Higher precision, lower recall

### Particle Classification
The system classifies detected particles into 5 types:
- **Fiber**: Thread-like particles
- **Fragment**: Broken pieces
- **Film**: Thin sheet-like particles
- **Pellet**: Small spherical particles
- **Foam**: Porous, bubble-like structures

### Polymer Identification
Simulated polymer type identification:
- PE (Polyethylene)
- PP (Polypropylene)
- PS (Polystyrene)
- PET (Polyethylene Terephthalate)
- PVC (Polyvinyl Chloride)
- PA (Polyamide/Nylon)
- Unknown

## Future Enhancements

### Planned Features
1. **Real LDIR Spectroscopy Integration**
2. **Database Storage for Results**
3. **Batch Processing Capabilities**
4. **Advanced Filtering and Search**
5. **Export to Scientific Formats**
6. **Model Performance Monitoring**
7. **User Authentication System**

### Model Improvements
1. **Multi-class Training** - Distinguish between particle types during training
2. **Size Classification** - Particle size categorization
3. **Shape Analysis** - Geometric feature extraction
4. **Confidence Calibration** - Better probability estimates
5. **Ensemble Methods** - Multiple model voting

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check if model files exist in correct paths
   - Verify file permissions
   - Ensure sufficient RAM for model loading

2. **CORS Errors**
   - Confirm both frontend and backend are running
   - Check allowed origins in Flask CORS configuration

3. **Image Processing Issues**
   - Validate image format support
   - Check file size limits
   - Verify OpenCV installation

4. **Performance Problems**
   - Monitor system resources
   - Consider GPU acceleration
   - Optimize batch processing

### Debugging Tools

**Backend Logging:**
```python
import logging
logging.basicConfig(level=logging.INFO)
```

**Frontend Debugging:**
```typescript
console.log('Detection results:', detectionResult);
```

**API Testing:**
```bash
curl -X POST http://localhost:5000/api/health
```

## Contributing

### Development Guidelines
1. Follow TypeScript and Python best practices
2. Maintain consistent code formatting
3. Write comprehensive tests
4. Document new features
5. Update this README with significant changes

### Testing
```bash
# Backend tests
python -m pytest test_*.py

# Frontend tests
npm test

# Integration tests
python test_api.py
```

---

*This documentation provides a comprehensive overview of the Microplastic Scout project architecture and implementation details. For specific implementation questions, refer to the individual component files and their inline documentation.*