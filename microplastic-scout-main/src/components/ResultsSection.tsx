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

  const handleExport = () => {
    const exportData = {
      imageName: result.imageName,
      timestamp: result.timestamp.toISOString(),
      mode: result.mode,
      processingTime: result.processingTime,
      totalCount: result.totalCount,
      countByType: result.countByType,
      detections: result.detections.map(det => ({
        id: det.id,
        particleType: det.particleType,
        polymerType: det.polymerType,
        confidence: det.confidence,
        ldirMatchScore: det.ldirMatchScore,
        boundingBox: det.boundingBox,
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
                {result.imageName} • Analyzed on {result.timestamp.toLocaleString()}
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
