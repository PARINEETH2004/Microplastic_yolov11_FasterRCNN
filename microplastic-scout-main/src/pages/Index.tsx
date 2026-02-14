import { useState, useCallback, useRef, useEffect } from 'react';
import { Header } from '@/components/Header';
import { HeroSection } from '@/components/HeroSection';
import { ImageUpload } from '@/components/ImageUpload';
import { ResultsSection } from '@/components/ResultsSection';
import { ProcessingOverlay } from '@/components/ProcessingOverlay';
import { Footer } from '@/components/Footer';
import { detectWithYolo } from '@/lib/yoloDetection';
import type { DetectionResult, DetectionMode } from '@/types/detection';

const Index = () => {
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [originalImage, setOriginalImage] = useState<string | null>(null); // Store original image URL
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingMode, setProcessingMode] = useState<DetectionMode>('fast');
  const [error, setError] = useState<string | null>(null);
  const [debugInfo, setDebugInfo] = useState<string[]>([]); // New debug state
  const uploadRef = useRef<HTMLDivElement>(null);

  const scrollToUpload = useCallback(() => {
    uploadRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const handleAnalyze = useCallback(async (file: File, mode: DetectionMode, algorithm: 'yolo' | 'faster_rcnn' = 'yolo') => {
    console.log('=== HANDLE ANALYZE CALLED ===');
    console.log('File received:', file.name, file.size, 'bytes');
    console.log('Mode:', mode);

    // Create object URL for the original image
    const imageUrl = URL.createObjectURL(file);
    setOriginalImage(imageUrl);

    // Add debug info to UI
    const newDebugInfo = [
      `=== ANALYSIS STARTED ===`,
      `File name: ${file.name}`,
      `File size: ${file.size} bytes`,
      `File type: ${file.type}`,
      `Detection mode: ${mode}`,
      `Timestamp: ${new Date().toLocaleTimeString()}`
    ];
    setDebugInfo(newDebugInfo);

    setIsProcessing(true);
    setProcessingMode(mode);
    setError(null);

    try {
      console.log('Starting analysis with mode:', mode);

      // Add FormData construction debug info
      const formData = new FormData();
      formData.append('image', file);
      formData.append('mode', mode);

      const imageField = formData.get('image');
      const modeField = formData.get('mode');

      const formDataDebug = [
        `FormData Construction:`,
        `image field exists: ${formData.has('image')}`,
        `mode field exists: ${formData.has('mode')}`,
        `image field type: ${imageField instanceof File ? 'File' : typeof imageField}`,
        `mode value: ${modeField}`,
        `File details in FormData: ${imageField instanceof File ? `${imageField.name} (${imageField.size} bytes)` : 'NOT A FILE'}`
      ];

      setDebugInfo(prev => [...prev, ...formDataDebug]);

      const detectionResult = await detectWithYolo(file, mode, algorithm);
      console.log('Analysis completed, result:', detectionResult);

      setDebugInfo(prev => [...prev, `✅ Analysis successful - ${detectionResult.totalCount} detections found`]);
      setResult(detectionResult);

      // Scroll to results section
      setTimeout(() => {
        document.querySelector('[data-results-section]')?.scrollIntoView({
          behavior: 'smooth'
        });
      }, 100);
    } catch (error) {
      console.error('Detection failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Analysis failed';
      setError(errorMessage);
      setDebugInfo(prev => [...prev, `❌ Analysis failed: ${errorMessage}`]);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const handleReset = useCallback(() => {
    setResult(null);
    setError(null);
    setDebugInfo([]);
    setOriginalImage(null); // Clear original image
  }, []);

  // Log state changes for debugging
  useEffect(() => {
    console.log('State updated - result:', result, 'isProcessing:', isProcessing, 'error:', error);
  }, [result, isProcessing, error]);

  // Cleanup object URLs when component unmounts
  useEffect(() => {
    return () => {
      if (originalImage) {
        URL.revokeObjectURL(originalImage);
      }
    };
  }, [originalImage]);

  return (
    <div className="min-h-screen bg-background">
      <Header />

      {!result ? (
        <>
          <HeroSection onScrollToUpload={scrollToUpload} />
          <div ref={uploadRef}>
            <ImageUpload onAnalyze={handleAnalyze} isProcessing={isProcessing} />
          </div>

          {/* Debug Information Display */}
          {debugInfo.length > 0 && (
            <div className="max-w-4xl mx-auto px-6 py-4">
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                <h3 className="font-semibold text-yellow-800 mb-2">Debug Information:</h3>
                <div className="text-sm text-yellow-700 font-mono">
                  {debugInfo.map((info, index) => (
                    <div key={index} className="mb-1">{info}</div>
                  ))}
                </div>
                <button
                  onClick={() => setDebugInfo([])}
                  className="mt-2 text-xs text-yellow-600 hover:text-yellow-800 underline"
                >
                  Clear Debug Info
                </button>
              </div>
            </div>
          )}

          {/* Error display */}
          {error && (
            <div className="max-w-4xl mx-auto px-6 py-4">
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
                <p className="text-destructive font-medium">Error: {error}</p>
                <button
                  onClick={() => setError(null)}
                  className="mt-2 text-sm text-destructive hover:underline"
                >
                  Dismiss
                </button>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="pt-16" data-results-section>
          <ResultsSection result={result} onReset={handleReset} originalImage={originalImage || undefined} />
        </div>
      )}

      <Footer />

      {isProcessing && <ProcessingOverlay mode={processingMode} />}
    </div>
  );
};

export default Index;