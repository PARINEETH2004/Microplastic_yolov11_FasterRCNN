import { useState, useCallback, useRef, useEffect } from 'react';
import { Header } from '@/components/Header';
import { HeroSection } from '@/components/HeroSection';
import { ImageUpload } from '@/components/ImageUpload';
import { ResultsSection } from '@/components/ResultsSection';
import { ProcessingOverlay } from '@/components/ProcessingOverlay';
import { Footer } from '@/components/Footer';
import { ErrorBoundary } from '@/components/ErrorBoundary';
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
    console.log('Algorithm:', algorithm);

    // Validate input
    if (!file || !file.name) {
      const error = 'Invalid file provided';
      console.error(error);
      setError(error);
      return;
    }

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

      // Validate the result structure
      if (!detectionResult || typeof detectionResult !== 'object') {
        throw new Error('Invalid result structure received from API');
      }

      if (!detectionResult.detections || !Array.isArray(detectionResult.detections)) {
        throw new Error('Invalid detections array in result');
      }

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
    console.log('=== STATE UPDATE ===');
    console.log('result:', result);
    console.log('result type:', typeof result);
    console.log('result is null:', result === null);
    console.log('result is undefined:', result === undefined);
    console.log('isProcessing:', isProcessing);
    console.log('error:', error);
    console.log('showing results section:', !!result);
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
          <div className="container mx-auto px-6 py-4 bg-green-100 border border-green-300 rounded-lg mb-4">
            <h2 className="text-lg font-semibold text-green-800">DEBUG: Results Section Active</h2>
            <p className="text-green-700">result is truthy: {String(!!result)}</p>
            <p className="text-green-700">result.detections exists: {String(!!result?.detections)}</p>
            <p className="text-green-700">detections length: {result?.detections?.length || 0}</p>
            
              
          </div>

          {/* Simple test component to verify rendering */}
          <div className="container mx-auto px-6 py-4 bg-yellow-100 border border-yellow-300 rounded-lg mb-4">
            <h3 className="text-lg font-semibold text-yellow-800">Simple Results Test</h3>
            <p className="text-yellow-700">Total detections: {result?.totalCount || 0}</p>
            <p className="text-yellow-700">Processing time: {result?.processingTime || 0}ms</p>
            <p className="text-yellow-700">Mode: {result?.mode || 'unknown'}</p>
          </div>

          {/* Try to render ResultsSection with error handling */}
          <div className="container mx-auto px-6">
            <h3 className="text-xl font-bold mb-4">Detailed Results:</h3>
            <div className="bg-white border rounded-lg p-4">
              {result ? (
                <ErrorBoundary>
                  <ResultsSection result={result} onReset={handleReset} originalImage={originalImage || undefined} />
                </ErrorBoundary>
              ) : (
                <div className="text-red-500 p-4">ERROR: No result data available to pass to ResultsSection</div>
              )}
            </div>
          </div>
        </div>
      )}

      <Footer />

      {isProcessing && <ProcessingOverlay mode={processingMode} />}
    </div>
  );
};

export default Index;