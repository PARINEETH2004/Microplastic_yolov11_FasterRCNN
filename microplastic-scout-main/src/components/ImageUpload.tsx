import { useState, useCallback, useEffect } from 'react';
import { Upload, Image as ImageIcon, X, Zap, Target, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { DetectionMode } from '@/types/detection';
import sampleImage from '@/assets/sample-microscopy.jpg';

interface ImageUploadProps {
  onAnalyze: (file: File, mode: DetectionMode, algorithm: 'yolo' | 'faster_rcnn') => void;
  isProcessing: boolean;
}

export function ImageUpload({ onAnalyze, isProcessing }: ImageUploadProps) {
  // CRITICAL: Explicit file state management
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [detectionMode, setDetectionMode] = useState<DetectionMode>('fast');
  const [detectionAlgorithm, setDetectionAlgorithm] = useState<'yolo' | 'faster_rcnn'>('yolo');
  const [isDragOver, setIsDragOver] = useState(false);

  // CRITICAL FIX 1: Remove circular dependency from handleFileSelect
  const handleFileSelect = useCallback((file: File) => {
    console.log('=== FILE SELECT TRIGGERED ===');
    console.log('File details:', {
      name: file.name,
      size: file.size,
      type: file.type,
      lastModified: file.lastModified
    });

    if (file.type.startsWith('image/')) {
      // CRITICAL: Store the actual File object in component state
      setSelectedFile(file);
      console.log('✅ File stored in local state:', file.name, file.size, 'bytes');

      // Create preview URL for display
      const blobUrl = URL.createObjectURL(file);
      setPreviewUrl(blobUrl);
      console.log('✅ Preview URL created:', blobUrl);

    } else {
      console.error('❌ Invalid file type:', file.type);
    }
  }, []); // Remove selectedFile dependency to avoid circular reference

  // CRITICAL FIX 2: Verify file state is properly maintained
  useEffect(() => {
    console.log('File state changed in ImageUpload:', selectedFile ? selectedFile.name : 'null');
  }, [selectedFile]);

  // CRITICAL FIX 3: Robust sample image loading with proper File creation
  const loadSampleImage = useCallback(async () => {
    console.log('=== LOADING SAMPLE IMAGE ===');
    try {
      const response = await fetch(sampleImage);
      if (!response.ok) {
        throw new Error(`Failed to fetch sample image: ${response.status}`);
      }

      const blob = await response.blob();
      console.log('Blob received:', {
        size: blob.size,
        type: blob.type
      });

      // CRITICAL: Create proper File object with correct metadata
      const file = new File([blob], 'sample-microscopy.jpg', {
        type: 'image/jpeg',
        lastModified: Date.now()
      });

      console.log('File object created:', {
        name: file.name,
        size: file.size,
        type: file.type,
        lastModified: file.lastModified
      });

      // Verify the file is valid before proceeding
      if (file.size === 0) {
        throw new Error('Sample image file is empty');
      }

      // CRITICAL: Explicitly call the file selection handler
      console.log('Calling handleFileSelect with created file...');
      handleFileSelect(file);

      // Additional verification - check state after a brief delay
      setTimeout(() => {
        console.log('Post-selection verification - local selectedFile:', selectedFile?.name || 'null');
        if (!selectedFile) {
          console.error('❌ CRITICAL ERROR: File was not stored in local state after handleFileSelect!');
        }
      }, 50);

    } catch (error) {
      console.error('❌ Failed to load sample image:', error);
    }
  }, [handleFileSelect]); // Only depend on handleFileSelect, not selectedFile

  // CRITICAL FIX 4: Enhanced analysis handler with explicit state verification
  const handleAnalyzeClick = useCallback(() => {
    console.log('=== START ANALYSIS CLICKED ===');
    console.log('Current local state:');
    console.log('- Selected file:', selectedFile ? selectedFile.name : 'null');
    console.log('- Selected file size:', selectedFile ? selectedFile.size : 'N/A');
    console.log('- Detection mode:', detectionMode);
    console.log('- isProcessing:', isProcessing);

    // CRITICAL: Verify file exists before proceeding
    if (!selectedFile) {
      console.error('❌ CRITICAL ERROR: No file selected for analysis!');
      console.error('This means the file was not properly stored in local state.');
      console.error('Check the file selection flow and ensure setSelectedFile() is called.');
      return;
    }

    console.log('✅ File found in local state, calling onAnalyze with:', selectedFile.name);
    console.log('File details being sent:', {
      name: selectedFile.name,
      size: selectedFile.size,
      type: selectedFile.type
    });

    // CRITICAL: Pass the actual File object to parent handler
    onAnalyze(selectedFile, detectionMode, detectionAlgorithm);
  }, [selectedFile, detectionMode, onAnalyze, isProcessing]);

  return (
    <section id="upload-section" className="py-16">
      <div className="container mx-auto px-6">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Section header */}
          <div className="text-center space-y-2">
            <h2 className="text-3xl font-bold text-foreground">Upload Image</h2>
            <p className="text-muted-foreground">
              Select a microscopy image (JPG, PNG) to analyze for microplastic particles
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Upload area */}
            <div
              // ... existing drag/drop handlers ...
              className={cn(
                "relative border-2 border-dashed rounded-xl p-8 transition-all duration-300 cursor-pointer",
                isDragOver
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50 hover:bg-muted/30",
                previewUrl && "border-solid border-primary/30"
              )}
            >
              {previewUrl ? (
                <div className="relative">
                  <img
                    src={previewUrl}
                    alt="Preview"
                    className="w-full h-64 object-contain rounded-lg bg-muted"
                  />
                  <button
                    onClick={() => {
                      setSelectedFile(null);
                      setPreviewUrl(null);
                      console.log('File selection cleared');
                    }}
                    className="absolute top-2 right-2 p-1.5 rounded-full bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-colors"
                  >
                    <X className="h-4 w-4" />
                  </button>
                  <div className="mt-4 text-center">
                    <p className="font-medium text-foreground">{selectedFile?.name}</p>
                    <p className="text-sm text-muted-foreground">
                      {selectedFile && (selectedFile.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                </div>
              ) : (
                <label className="flex flex-col items-center justify-center h-64 cursor-pointer">
                  <div className="p-4 rounded-full bg-primary/10 mb-4">
                    <Upload className="h-8 w-8 text-primary" />
                  </div>
                  <p className="text-lg font-medium text-foreground mb-1">
                    Drop your image here
                  </p>
                  <p className="text-sm text-muted-foreground mb-4">
                    or click to browse
                  </p>
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <ImageIcon className="h-4 w-4" />
                    Supports JPG, PNG up to 10MB
                  </div>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleFileSelect(file);
                    }}
                    className="hidden"
                  />
                </label>
              )}

              {/* Sample image button */}
              {!previewUrl && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2">
                  <Button
                    variant="glass"
                    size="sm"
                    onClick={loadSampleImage}
                    className="gap-2"
                  >
                    <Sparkles className="h-4 w-4" />
                    Try sample image
                  </Button>
                </div>
              )}
            </div>

            {/* Detection mode selection */}
            <div className="space-y-6">
              {/* Detection mode selection */}
              <div className="space-y-4">
                <h3 className="font-medium text-foreground">Detection Mode</h3>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setDetectionMode('fast')}
                    className={cn(
                      "flex flex-col items-center gap-2 p-4 rounded-lg border transition-all",
                      detectionMode === 'fast'
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:bg-muted/50'
                    )}
                  >
                    <Zap className="h-5 w-5 text-primary" />
                    <span className="text-sm font-medium">Fast</span>
                    <span className="text-xs text-muted-foreground">~1.5s</span>
                  </button>
                  <button
                    onClick={() => setDetectionMode('accurate')}
                    className={cn(
                      "flex flex-col items-center gap-2 p-4 rounded-lg border transition-all",
                      detectionMode === 'accurate'
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:bg-muted/50'
                    )}
                  >
                    <Target className="h-5 w-5 text-primary" />
                    <span className="text-sm font-medium">Accurate</span>
                    <span className="text-xs text-muted-foreground">~3s</span>
                  </button>
                </div>
              </div>

              {/* Algorithm selection */}
              <div className="space-y-4">
                <h3 className="font-medium text-foreground">Detection Algorithm</h3>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    onClick={() => setDetectionAlgorithm('yolo')}
                    className={cn(
                      "flex flex-col items-center gap-2 p-4 rounded-lg border transition-all",
                      detectionAlgorithm === 'yolo'
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:bg-muted/50'
                    )}
                  >
                    <span className="h-5 w-5 font-bold text-primary">Y</span>
                    <span className="text-sm font-medium">YOLOv11</span>
                    <span className="text-xs text-muted-foreground">Real-time</span>
                  </button>
                  <button
                    onClick={() => setDetectionAlgorithm('faster_rcnn')}
                    className={cn(
                      "flex flex-col items-center gap-2 p-4 rounded-lg border transition-all",
                      detectionAlgorithm === 'faster_rcnn'
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:bg-muted/50'
                    )}
                  >
                    <span className="h-5 w-5 font-bold text-primary">R</span>
                    <span className="text-sm font-medium">Faster R-CNN</span>
                    <span className="text-xs text-muted-foreground">High accuracy</span>
                  </button>
                </div>
              </div>

              {/* CRITICAL: Use the enhanced analysis handler */}
              <Button
                variant="hero"
                size="lg"
                onClick={handleAnalyzeClick}
                disabled={!selectedFile || isProcessing}
                className="w-full"
              >
                {isProcessing ? (
                  <>
                    <div className="h-5 w-5 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    Start Analysis
                  </>
                )}
              </Button>

              {/* Info */}
              <p className="text-xs text-center text-muted-foreground">
                Detection includes simulated LDIR spectroscopy for polymer identification
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}