import { useState } from 'react';
import { Download, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { DetectionOverlay } from '@/components/DetectionOverlay';
import { SpectrumViewer } from '@/components/SpectrumViewer';
import { ResultsTable } from '@/components/ResultsTable';
import { StatsSummary } from '@/components/StatsSummary';
import type { Detection, DetectionResult } from '@/types/detection';

interface ResultsSectionProps {
  result: DetectionResult;
  onReset: () => void;
  originalImage?: string; // Add prop for original image
}

export function ResultsSection({ result, onReset, originalImage }: ResultsSectionProps) {
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);

  // Error boundary for component rendering
  if (!result) {
    return (
      <div className="container mx-auto px-6 py-16 text-center">
        <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-8 max-w-2xl mx-auto">
          <h3 className="text-xl font-semibold text-destructive mb-2">Error: Missing Results Data</h3>
          <p className="text-destructive mb-4">The analysis results could not be loaded properly.</p>
          <Button onClick={onReset} variant="outline">Try Again</Button>
        </div>
      </div>
    );
  }

  // Additional validation
  if (!result.detections || !Array.isArray(result.detections)) {
    return (
      <div className="container mx-auto px-6 py-16 text-center">
        <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-8 max-w-2xl mx-auto">
          <h3 className="text-xl font-semibold text-destructive mb-2">Error: Invalid Detection Data</h3>
          <p className="text-destructive mb-4">The detection results are malformed.</p>
          <Button onClick={onReset} variant="outline">Try Again</Button>
        </div>
      </div>
    );
  }

  const handleExport = () => {
    const exportData = {
      imageName: result.imageName || 'Unknown',
      timestamp: (result.timestamp || new Date()).toISOString(),
      mode: result.mode || 'unknown',
      processingTime: result.processingTime || 0,
      totalCount: result.totalCount || 0,
      countByType: result.countByType || {},
      detections: (result.detections || []).map(det => ({
        id: det.id || '',
        particleType: det.particleType || 'Unknown',
        polymerType: det.polymerType || 'Unknown',
        confidence: det.confidence || 0,
        ldirMatchScore: det.ldirMatchScore || 0,
        boundingBox: det.boundingBox || { x: 0, y: 0, width: 0, height: 0 },
      })),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `microplastic-analysis-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section className="py-16 bg-muted/30">
      <div className="container mx-auto px-6">
        <div className="max-w-7xl mx-auto space-y-8">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h2 className="text-3xl font-bold text-foreground">Analysis Results</h2>
              <p className="text-muted-foreground">
                {result.imageName || 'Unknown'} • Analyzed on {(result.timestamp || new Date()).toLocaleString()}
              </p>
            </div>
            <div className="flex gap-3">
              <Button variant="outline" onClick={onReset}>
                <RefreshCw className="h-4 w-4" />
                New Analysis
              </Button>
              <Button onClick={handleExport}>
                <Download className="h-4 w-4" />
                Export JSON
              </Button>
            </div>
          </div>

          {/* Stats summary */}
          <StatsSummary result={result} />

          {/* Main content grid */}
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Image with overlays */}
            <div className="space-y-4">
              <h3 className="font-semibold text-foreground">Detection Visualization</h3>
              <DetectionOverlay
                result={result}
                selectedDetection={selectedDetection}
                onSelectDetection={setSelectedDetection}
                originalImage={originalImage}
              />
            </div>

            {/* Spectrum viewer or placeholder */}
            <div className="space-y-4">
              <h3 className="font-semibold text-foreground">LDIR Analysis</h3>
              {selectedDetection ? (
                <SpectrumViewer detection={selectedDetection} />
              ) : (
                <div className="bg-card rounded-xl border border-border p-8 h-[300px] flex items-center justify-center">
                  <div className="text-center text-muted-foreground">
                    <p className="text-lg font-medium">Select a Detection</p>
                    <p className="text-sm mt-1">
                      Click on a bounding box or table row to view the simulated LDIR spectrum
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Results table */}
          <ResultsTable
            result={result}
            selectedDetection={selectedDetection}
            onSelectDetection={setSelectedDetection}
          />
        </div>
      </div>
    </section>
  );
}
