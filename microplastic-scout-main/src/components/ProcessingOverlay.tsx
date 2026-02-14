import { useEffect, useState } from 'react';
import type { DetectionMode } from '@/types/detection';

interface ProcessingOverlayProps {
  mode: DetectionMode;
}

const steps = [
  { id: 1, label: 'Image Preprocessing', duration: 500 },
  { id: 2, label: 'YOLOv11 Detection', duration: 1000 },
  { id: 3, label: 'Faster R-CNN Refinement', duration: 1500, accurateOnly: true },
  { id: 4, label: 'LDIR Spectroscopy Simulation', duration: 800 },
  { id: 5, label: 'Result Aggregation', duration: 400 },
];

export function ProcessingOverlay({ mode }: ProcessingOverlayProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);

  const activeSteps = steps.filter(s => !s.accurateOnly || mode === 'accurate');

  useEffect(() => {
    let stepIndex = 0;
    let accumulated = 0;
    const totalDuration = activeSteps.reduce((sum, s) => sum + s.duration, 0);

    const interval = setInterval(() => {
      accumulated += 50;
      setProgress((accumulated / totalDuration) * 100);

      let accumulatedDuration = 0;
      for (let i = 0; i < activeSteps.length; i++) {
        accumulatedDuration += activeSteps[i].duration;
        if (accumulated < accumulatedDuration) {
          setCurrentStep(i);
          break;
        }
      }
    }, 50);

    return () => clearInterval(interval);
  }, [mode]);

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center">
      <div className="bg-card rounded-2xl border border-border p-8 max-w-md w-full mx-4 shadow-lg">
        <div className="text-center mb-8">
          <h3 className="text-xl font-semibold text-foreground mb-2">Processing Image</h3>
          <p className="text-sm text-muted-foreground">
            {mode === 'accurate' ? 'Running YOLOv11 + Faster R-CNN pipeline' : 'Running YOLOv11 fast detection'}
          </p>
        </div>

        {/* Progress bar */}
        <div className="h-2 bg-muted rounded-full overflow-hidden mb-6">
          <div 
            className="h-full bg-primary transition-all duration-100 ease-linear"
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>

        {/* Steps */}
        <div className="space-y-3">
          {activeSteps.map((step, index) => (
            <div 
              key={step.id}
              className={`flex items-center gap-3 transition-opacity ${
                index <= currentStep ? 'opacity-100' : 'opacity-40'
              }`}
            >
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${
                index < currentStep 
                  ? 'bg-primary text-primary-foreground' 
                  : index === currentStep 
                    ? 'bg-primary/20 text-primary border-2 border-primary animate-pulse'
                    : 'bg-muted text-muted-foreground'
              }`}>
                {index < currentStep ? '✓' : index + 1}
              </div>
              <span className={`text-sm ${
                index === currentStep ? 'text-foreground font-medium' : 'text-muted-foreground'
              }`}>
                {step.label}
              </span>
              {index === currentStep && (
                <div className="ml-auto">
                  <div className="h-4 w-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
