# Microplastic Scout - Developer Guide

## Key Implementation Details

### 1. YOLOv11 Integration

**Model Loading Priority:**
```python
# backend/detection.py - Lines 22-39
def _load_model(self):
    # Priority 1: Custom trained model
    trained_model_path = 'runs/detect/microplastic_v11/weights/best.pt'
    
    # Priority 2: Custom model file  
    custom_model_path = 'models/yolov11_microplastic.pt'
    
    # Priority 3: Pretrained model fallback
    pretrained_model = 'yolov11s.pt'
```

**Detection Process:**
```python
# backend/detection.py - Lines 105-144
def detect_microplastics(self, image_bytes: bytes, mode: str):
    # 1. Convert image bytes to OpenCV format
    img = self._preprocess_image(image_bytes)
    
    # 2. Run YOLO inference with mode-specific thresholds
    conf_threshold = 0.3 if mode == 'fast' else 0.25
    results = self.model(img, conf=conf_threshold, iou=0.45)
    
    # 3. Process detection results
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract coordinates and confidence
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            
            # Classify particle type based on confidence
            particle_type = self._classify_particle_type(confidence, class_id)
            polymer_type = self._classify_polymer_type()
```

### 2. Frontend-Backend Communication

**API Request Flow:**
```typescript
// src/lib/yoloDetection.ts - Lines 70-99
async function detectWithYolo(imageFile: File, mode: DetectionMode) {
    const formData = new FormData();
    formData.append('image', imageFile);  // Image file
    formData.append('mode', mode);         // Detection mode
    
    const response = await fetch('/api/detect', {
        method: 'POST',
        body: formData  // No Content-Type header (browser sets automatically)
    });
    
    return response.json();  // DetectionResult object
}
```

**Backend Response Handling:**
```python
# backend/app.py - Lines 50-158
@app.route('/api/detect', methods=['POST'])
def detect_microplastics():
    # Extract image and mode from request
    image_file = request.files['image']
    mode = request.form.get('mode', 'fast')
    
    # Process image
    image_bytes = image_file.read()
    results = detector.detect_microplastics(image_bytes, mode)
    
    # Return structured response
    return jsonify({
        'imageUrl': f'/api/images/{image_name}',
        'imageName': filename,
        'timestamp': time.time(),
        'mode': mode,
        'processingTime': int(processing_time * 1000),
        'detections': results['detections'],
        'totalCount': results['totalCount'],
        'countByType': results['countByType']
    })
```

### 3. Image Visualization System

**Original Image Handling:**
```typescript
// src/pages/Index.tsx - Lines 26-29
const handleAnalyze = useCallback(async (file: File, mode: DetectionMode) => {
    // Create object URL for original image
    const imageUrl = URL.createObjectURL(file);
    setOriginalImage(imageUrl);  // Store for visualization
    
    // ... detection logic
}, []);
```

**Detection Overlay Component:**
```typescript
// src/components/DetectionOverlay.tsx - Lines 22-30
useEffect(() => {
    // Use original image if available, fallback to backend URL
    if (originalImage) {
        setImageSrc(originalImage);
    } else {
        setImageSrc(result.imageUrl);  // Backend fallback
    }
}, [originalImage, result.imageUrl]);
```

**SVG Bounding Box Rendering:**
```typescript
// src/components/DetectionOverlay.tsx - Lines 34-79
<svg viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}>
    {result.detections.map((det) => (
        <g key={det.id}>
            {/* Bounding box rectangle */}
            <rect
                x={det.boundingBox.x}
                y={det.boundingBox.y}
                width={det.boundingBox.width}
                height={det.boundingBox.height}
                stroke={getParticleColor(det.particleType)}
                strokeWidth={selectedDetection?.id === det.id ? 3 : 2}
            />
            
            {/* Label background and text */}
            <rect x={det.boundingBox.x} y={det.boundingBox.y - 20} ... />
            <text x={det.boundingBox.x + 4} y={det.boundingBox.y - 6} ...>
                {det.particleType} {(det.confidence * 100).toFixed(0)}%
            </text>
        </g>
    ))}
</svg>
```

### 4. Type Safety and Data Structures

**TypeScript Interfaces:**
```typescript
// src/types/detection.ts
export interface Detection {
    id: string;
    particleType: ParticleType;        // 'fiber' | 'fragment' | 'film' | 'pellet' | 'foam'
    polymerType: PolymerType;          // 'PE' | 'PP' | 'PS' | 'PET' | 'PVC' | 'PA' | 'Unknown'
    confidence: number;                // 0.0 - 1.0
    boundingBox: BoundingBox;          // {x, y, width, height}
    ldirMatchScore: number;            // Simulated LDIR match score
    spectrumData: number[];            // IR spectrum simulation
}

export interface DetectionResult {
    imageUrl: string;
    imageName: string;
    timestamp: Date;
    mode: 'fast' | 'accurate';
    processingTime: number;
    detections: Detection[];
    totalCount: number;
    countByType: Record<ParticleType, number>;
}
```

### 5. Configuration and Constants

**Backend Configuration:**
```python
# backend/config.py
class Config:
    SECRET_KEY = 'microplastic-scout-secret-key'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    MODEL_PATH = 'models/yolov11_microplastic.pt'
    CONFIDENCE_THRESHOLD = 0.5
    IMAGE_SIZE = 640
    CLASS_NAMES = ['Microplastic']
    PARTICLE_TYPES = ['fiber', 'fragment', 'film', 'pellet', 'foam']
    POLYMER_TYPES = ['PE', 'PP', 'PS', 'PET', 'PVC', 'PA', 'Unknown']
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
```

**Frontend Constants:**
```typescript
// src/lib/yoloDetection.ts
const API_BASE_URL = '/api';  // Uses Vite proxy in development

// src/lib/mockDetection.ts
const particleTypes: ParticleType[] = ['fiber', 'fragment', 'film', 'pellet', 'foam'];
const polymerTypes: PolymerType[] = ['PE', 'PP', 'PS', 'PET', 'PVC', 'PA'];
```

### 6. Error Handling and Fallbacks

**Backend Error Handling:**
```python
# backend/app.py - Lines 156-158
except Exception as e:
    logger.error(f"Detection error: {str(e)}", exc_info=True)
    return jsonify({'error': f'Detection failed: {str(e)}'}), 500
```

**Frontend Fallback Logic:**
```typescript
// src/lib/yoloDetection.ts - Lines 142-159
try {
    const isAvailable = await yoloDetectionService.isBackendAvailable();
    if (isAvailable) {
        return yoloDetectionService.detectMicroplastics(imageFile, mode);
    } else {
        // Fallback to mock detection when backend unavailable
        const { simulateDetection } = await import('./mockDetection');
        return simulateDetection(URL.createObjectURL(imageFile), imageFile.name, mode);
    }
} catch (error) {
    // Always fallback to mock detection on any error
    const { simulateDetection } = await import('./mockDetection');
    return simulateDetection(URL.createObjectURL(imageFile), imageFile.name, mode);
}
```

### 7. Performance Optimizations

**Image Processing:**
```python
# backend/detection.py - Lines 51-59
def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
    # Efficient conversion from bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img
```

**Frontend Optimization:**
```typescript
// src/pages/Index.tsx - Lines 94-96
useEffect(() => {
    return () => {
        if (originalImage) {
            URL.revokeObjectURL(originalImage);  // Memory cleanup
        }
    };
}, [originalImage]);
```

### 8. Testing and Debugging

**Backend Testing:**
```python
# backend/test_api.py
def test_detection_endpoint():
    # Test image upload and detection
    with open('test_image.jpg', 'rb') as f:
        response = client.post('/api/detect', 
                             data={'image': f, 'mode': 'fast'})
        assert response.status_code == 200
```

**Frontend Debugging:**
```typescript
// src/pages/Index.tsx - Lines 24-37
const newDebugInfo = [
    `=== ANALYSIS STARTED ===`,
    `File name: ${file.name}`,
    `File size: ${file.size} bytes`,
    `Detection mode: ${mode}`,
    `Timestamp: ${new Date().toLocaleTimeString()}`
];
setDebugInfo(newDebugInfo);  // Display in UI for debugging
```

## Key Integration Points

### 1. Model → Backend → Frontend Flow
```
YOLOv11 Model (best.pt)
    ↓
detection.py (MicroplasticDetector class)
    ↓
app.py (Flask API endpoints)  
    ↓
yoloDetection.ts (Frontend API client)
    ↓
DetectionOverlay.tsx (Image visualization)
```

### 2. Data Transformation Chain
```
Image File Upload
    ↓
FormData (multipart/form-data)
    ↓
Flask Request Processing
    ↓
OpenCV Image Processing
    ↓
YOLOv11 Inference
    ↓
Detection Results (Python dict)
    ↓
JSON Response
    ↓
TypeScript DetectionResult
    ↓
React Component Props
    ↓
SVG Visualization
```

### 3. Error Handling Cascade
```
Model Loading Failure
    ↓
Pretrained Model Fallback
    ↓
Backend Connection Error
    ↓
Mock Detection Fallback
    ↓
User-friendly Error Message
```

This guide provides the essential implementation details needed to understand, modify, and extend the Microplastic Scout application.