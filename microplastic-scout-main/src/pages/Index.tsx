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
  const uploadRef = useRef<HTMLDivElement>(null);

  const scrollToUpload = useCallback(() => {
    uploadRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  const handleAnalyze = useCallback(async (file: File, mode: DetectionMode, algorithm: 'yolo' | 'faster_rcnn' = 'yolo') => {
    // Validate input
    if (!file || !file.name) {
      const error = 'Invalid file provided';
      setError(error);
      return;
    }

    // Create object URL for the original image
    const imageUrl = URL.createObjectURL(file);
    setOriginalImage(imageUrl);

    setIsProcessing(true);
    setProcessingMode(mode);
    setError(null);

    try {
      const detectionResult = await detectWithYolo(file, mode, algorithm);

      // Validate the result structure
      if (!detectionResult || typeof detectionResult !== 'object') {
        throw new Error('Invalid result structure received from API');
      }

      if (!detectionResult.detections || !Array.isArray(detectionResult.detections)) {
        throw new Error('Invalid detections array in result');
      }

      setResult(detectionResult);

      // Scroll to results section
      setTimeout(() => {
        document.querySelector('[data-results-section]')?.scrollIntoView({
          behavior: 'smooth'
        });
      }, 100);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Analysis failed';
      setError(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const handleReset = useCallback(() => {
    setResult(null);
    setError(null);
    setOriginalImage(null); // Clear original image
  }, []);

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