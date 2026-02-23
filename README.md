# Microplastic Scout

A full-stack application for automated microplastic particle detection and classification using both YOLOv11 and Faster R-CNN deep learning models.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Running the Application

**Backend (Python Flask):**
```bash
cd backend
pip install -r requirements.txt
python app.py
# Server starts on http://localhost:5000
```

**Frontend (React):**
```bash
cd microplastic-scout-main
npm install
npm run dev
# Server starts on http://localhost:8081
```

## 📋 Project Overview

Microplastic Scout is designed to help researchers and environmental scientists automatically detect and classify microplastic particles in environmental samples. The system combines state-of-the-art computer vision with an intuitive web interface.

### Key Features
- **Real-time Detection**: Upload images and get instant microplastic detection results
- **Multiple Detection Modes**: Fast processing or high-accuracy analysis
- **Dual Algorithm Support**: Choose between YOLOv11 (speed) and Faster R-CNN (accuracy)
- **Particle Classification**: Automatic categorization into fiber, fragment, film, pellet, or foam
- **Polymer Identification**: Simulated polymer type detection (PE, PP, PS, PET, PVC, PA)
- **Interactive Visualization**: Bounding boxes overlaid on original images
- **Result Export**: Download analysis results in JSON format

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    HTTP/REST    ┌──────────────────┐
│   React Frontend│ ←──────────────→ │  Flask Backend   │
│   (Vite/TS)     │                 │   (Python)       │
└─────────────────┘                 └──────────────────┘
                                            │
                                            ▼
                                  ┌──────────────────┐
                                  │  Detection Models │
                                  │  (YOLOv11 &       │
                                  │   Faster R-CNN)   │
                                  └──────────────────┘
```

### Detailed Architecture Documentation
For comprehensive technical details about the system architecture, component connections, and dual model implementation (YOLOv11 & Faster R-CNN), see: [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)

## 🔬 Dual Model Implementation

### Model Details
- **YOLOv11 Model**: YOLOv11s (pretrained) with custom microplastic detection training
- **Faster R-CNN Model**: ResNet50 backbone with custom microplastic detection training
- **Input Size**: 640×640 pixels
- **Confidence Thresholds**: 
  - Fast mode: 0.3
  - Accurate mode: 0.25

### Model Files Structure
```
backend/
├── models/
│   └── yolov11_microplastic.pt      # Custom trained YOLOv11 model
├── faster_rcnn_detector.py          # Faster R-CNN implementation
├── runs/
│   └── detect/
│       └── microplastic_v11/
│           └── weights/
│               ├── best.pt          # Best YOLOv11 weights
│               └── last.pt          # Last YOLOv11 epoch
├── yolov11s.pt                      # Pretrained base model
└── detection.py                     # Model integration
```

### Detection Pipeline
1. **Image Preprocessing**: Convert to OpenCV format
2. **Model Inference**: Run either YOLOv11 or Faster R-CNN detection
3. **Post-processing**: Filter by confidence threshold
4. **Classification**: Particle and polymer type identification
5. **Result Formatting**: JSON response with bounding boxes

## 🛠️ Technology Stack

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **shadcn/ui** - Component library
- **React Query** - State management

### Backend
- **Python 3.8+** - Core language
- **Flask** - Web framework
- **Ultralytics YOLO** - YOLOv11 computer vision
- **TorchVision** - Faster R-CNN computer vision
- **OpenCV** - Image processing
- **PyTorch** - Deep learning
- **NumPy** - Numerical computing

## 📊 Data Flow

### Request Processing
```
User Upload → Frontend → FormData → Backend API → YOLO Model → Results → Visualization
```

### Response Format
```json
{
  "imageUrl": "/api/images/uuid_filename.jpg",
  "imageName": "sample.jpg",
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
      "boundingBox": {"x": 120, "y": 85, "width": 45, "height": 120}
    }
  ],
  "countByType": {"fiber": 12, "fragment": 8, "film": 2, "pellet": 1, "foam": 0}
}
```

## 🎯 Usage

1. **Upload Image**: Select a microplastic sample image (JPG, PNG, etc.)
2. **Choose Mode**: 
   - Fast: Quick detection (~1.5s)
   - Accurate: Thorough analysis (~3.0s)
3. **Select Algorithm**: 
   - YOLOv11: Faster processing
   - Faster R-CNN: Higher accuracy
4. **View Results**: 
   - Original image with bounding boxes
   - Particle type classifications
   - Confidence scores
   - Count statistics
5. **Export Data**: Download results as JSON

## 📁 Project Structure

```
microplastic-scout-main/
├── backend/                    # Python Flask backend
│   ├── app.py                 # Main Flask application
│   ├── detection.py           # YOLO detection logic
│   ├── config.py              # Configuration settings
│   ├── models/                # Model files
│   ├── runs/                  # Training outputs
│   └── requirements.txt       # Python dependencies
├── microplastic-scout-main/   # React frontend
│   ├── src/                   # Source code
│   │   ├── components/        # React components
│   │   ├── lib/               # Utility functions
│   │   ├── pages/             # Page components
│   │   └── types/             # TypeScript types
│   └── package.json           # Node.js dependencies
├── microplastic images/       # Training dataset
│   ├── train/                 # Training data
│   ├── valid/                 # Validation data
│   └── test/                  # Test data
└── PROJECT_ARCHITECTURE.md    # Detailed architecture docs
```

## 🔧 Development

### Backend Development
```bash
cd backend
# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py

# Run tests
python -m pytest
```

### Frontend Development
```bash
cd microplastic-scout-main
# Install dependencies
npm install

# Run development server
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

## 🐛 Troubleshooting

### Common Issues

1. **Port Conflicts**
   - Backend: http://localhost:5000
   - Frontend: http://localhost:8081

2. **Model Loading Errors**
   - Check model file paths
   - Verify file permissions
   - Ensure sufficient RAM

3. **CORS Issues**
   - Confirm both servers are running
   - Check Flask CORS configuration

4. **Image Display Problems**
   - Verify image format support
   - Check file size limits
   - Review browser console for errors

### Debug Information
Enable detailed logging:
```python
# backend/app.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

```typescript
// frontend console
console.log('Debug info:', detectionResult);
```

## 📚 Documentation

- **Architecture Details**: [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)
- **API Documentation**: See backend/app.py endpoints
- **Component Documentation**: Inline comments in source files
- **Type Definitions**: src/types/detection.ts

## 🤝 Contributing

### Guidelines
1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Submit pull requests with clear descriptions

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Run tests
5. Submit pull request

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- **Ultralytics** for YOLOv11 implementation
- **PyTorch/TorchVision** for Faster R-CNN implementation
- **React** and **Vite** teams for excellent development tools
- **Flask** community for the web framework
- Environmental research community for the important work

## 📞 Support

For issues, questions, or contributions:
- Check the detailed architecture documentation
- Review existing issues
- Submit new issues with detailed reproduction steps

---

*For comprehensive technical implementation details, refer to [PROJECT_ARCHITECTURE.md](PROJECT_ARCHITECTURE.md)*
