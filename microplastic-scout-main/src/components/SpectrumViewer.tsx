import { useMemo } from 'react';
import type { Detection } from '@/types/detection';
import { getPolymerFullName } from '@/lib/mockDetection';

interface SpectrumViewerProps {
  detection: Detection;
}

export function SpectrumViewer({ detection }: SpectrumViewerProps) {
  const spectrumPath = useMemo(() => {
    const data = detection.spectrumData;
    const width = 300;
    const height = 100;
    const stepX = width / (data.length - 1);
    
    let path = `M 0 ${height - data[0] * height}`;
    for (let i = 1; i < data.length; i++) {
      path += ` L ${i * stepX} ${height - data[i] * height}`;
    }
    return path;
  }, [detection.spectrumData]);

  return (
    <div className="bg-card rounded-xl border border-border p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-foreground">Simulated LDIR Spectrum</h3>
        <div className="px-3 py-1 rounded-full bg-accent/10 text-accent text-sm font-medium">
          {detection.polymerType}
        </div>
      </div>
      
      {/* Spectrum visualization */}
      <div className="relative bg-muted/50 rounded-lg p-4">
        <svg viewBox="0 0 300 100" className="w-full h-32">
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map((y) => (
            <line
              key={y}
              x1="0"
              y1={y}
              x2="300"
              y2={y}
              stroke="hsl(var(--border))"
              strokeWidth="0.5"
              strokeDasharray="4 4"
            />
          ))}
          
          {/* Spectrum line */}
          <path
            d={spectrumPath}
            fill="none"
            stroke="hsl(var(--primary))"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          
          {/* Gradient fill */}
          <defs>
            <linearGradient id="spectrumGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity="0.3" />
              <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity="0" />
            </linearGradient>
          </defs>
          <path
            d={`${spectrumPath} L 300 100 L 0 100 Z`}
            fill="url(#spectrumGradient)"
          />
        </svg>
        
        {/* X-axis label */}
        <div className="flex justify-between text-xs text-muted-foreground mt-2">
          <span>4000 cm⁻¹</span>
          <span>Wavenumber</span>
          <span>400 cm⁻¹</span>
        </div>
      </div>
      
      {/* Detection details */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground uppercase tracking-wide">Polymer Type</p>
          <p className="font-medium text-foreground">{getPolymerFullName(detection.polymerType)}</p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground uppercase tracking-wide">Match Score</p>
          <p className="font-medium text-foreground">{(detection.ldirMatchScore * 100).toFixed(1)}%</p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground uppercase tracking-wide">Particle Type</p>
          <p className="font-medium text-foreground capitalize">{detection.particleType}</p>
        </div>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground uppercase tracking-wide">Confidence</p>
          <p className="font-medium text-foreground">{(detection.confidence * 100).toFixed(1)}%</p>
        </div>
      </div>
      
      {/* Note */}
      <p className="text-xs text-muted-foreground bg-muted/50 rounded-lg p-3">
        ⚠️ This is a <strong>simulated</strong> LDIR spectrum for demonstration purposes. 
        Phase-2 will integrate real spectroscopy hardware for accurate polymer identification.
      </p>
    </div>
  );
}
