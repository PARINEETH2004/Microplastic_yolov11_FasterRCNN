import type { DetectionResult } from '@/types/detection';
import { getParticleColor } from '@/lib/mockDetection';
import { Clock, Target, Zap, FlaskConical } from 'lucide-react';

interface StatsSummaryProps {
  result: DetectionResult;
}

export function StatsSummary({ result }: StatsSummaryProps) {
  const particleTypes = Object.entries(result.countByType).filter(([, count]) => count > 0);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Total count */}
      <div className="bg-card rounded-xl border border-border p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Target className="h-5 w-5 text-primary" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">{result.totalCount}</p>
            <p className="text-sm text-muted-foreground">Particles Detected</p>
          </div>
        </div>
      </div>

      {/* Processing time */}
      <div className="bg-card rounded-xl border border-border p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-accent/10">
            <Clock className="h-5 w-5 text-accent" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">
              {(result.processingTime / 1000).toFixed(2)}s
            </p>
            <p className="text-sm text-muted-foreground">Processing Time</p>
          </div>
        </div>
      </div>

      {/* Detection mode */}
      <div className="bg-card rounded-xl border border-border p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-secondary">
            {result.mode === 'fast' ? (
              <Zap className="h-5 w-5 text-secondary-foreground" />
            ) : (
              <Target className="h-5 w-5 text-secondary-foreground" />
            )}
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground capitalize">{result.mode}</p>
            <p className="text-sm text-muted-foreground">Detection Mode</p>
          </div>
        </div>
      </div>

      {/* Detection algorithm */}
      <div className="bg-card rounded-xl border border-border p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-secondary">
            {result.detections.some(d => d.algorithm === 'faster_rcnn') ? (
              <span className="text-lg font-bold text-secondary-foreground">R</span>
            ) : (
              <span className="text-lg font-bold text-secondary-foreground">Y</span>
            )}
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">
              {result.detections.some(d => d.algorithm === 'faster_rcnn') ? 'Faster R-CNN' : 'YOLOv11'}
            </p>
            <p className="text-sm text-muted-foreground">Algorithm</p>
          </div>
        </div>
      </div>

      {/* Polymer types */}
      <div className="bg-card rounded-xl border border-border p-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-fragment/10">
            <FlaskConical className="h-5 w-5 text-fragment" />
          </div>
          <div>
            <p className="text-2xl font-bold text-foreground">
              {new Set(result.detections.map(d => d.polymerType)).size}
            </p>
            <p className="text-sm text-muted-foreground">Polymer Types</p>
          </div>
        </div>
      </div>

      {/* Particle type breakdown */}
      <div className="col-span-2 lg:col-span-4 bg-card rounded-xl border border-border p-4">
        <h4 className="font-medium text-foreground mb-4">Particle Distribution</h4>
        <div className="flex flex-wrap gap-3">
          {particleTypes.map(([type, count]) => (
            <div
              key={type}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted/50"
            >
              <div
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: getParticleColor(type as any) }}
              />
              <span className="font-medium capitalize text-foreground">{type}</span>
              <span className="text-muted-foreground">({count})</span>
            </div>
          ))}
        </div>

        {/* Visual bar chart */}
        <div className="mt-4 flex h-4 rounded-full overflow-hidden bg-muted">
          {particleTypes.map(([type, count]) => (
            <div
              key={type}
              className="h-full transition-all"
              style={{
                backgroundColor: getParticleColor(type as any),
                width: `${(count / result.totalCount) * 100}%`
              }}
              title={`${type}: ${count} (${((count / result.totalCount) * 100).toFixed(1)}%)`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
