import type { Detection, DetectionResult } from '@/types/detection';
import { getParticleColor, getPolymerFullName } from '@/lib/mockDetection';
import { cn } from '@/lib/utils';

interface ResultsTableProps {
  result: DetectionResult;
  selectedDetection: Detection | null;
  onSelectDetection: (detection: Detection | null) => void;
}

export function ResultsTable({ 
  result, 
  selectedDetection, 
  onSelectDetection 
}: ResultsTableProps) {
  return (
    <div className="bg-card rounded-xl border border-border overflow-hidden">
      <div className="p-4 border-b border-border">
        <h3 className="font-semibold text-foreground">Detection Results</h3>
        <p className="text-sm text-muted-foreground">
          {result.totalCount} particles detected in {(result.processingTime / 1000).toFixed(2)}s
        </p>
      </div>
      
      <div className="max-h-[400px] overflow-y-auto">
        <table className="w-full">
          <thead className="bg-muted/50 sticky top-0">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                ID
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Type
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Polymer
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Confidence
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">
                LDIR Match
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {result.detections.map((det, index) => (
              <tr
                key={det.id}
                onClick={() => onSelectDetection(selectedDetection?.id === det.id ? null : det)}
                className={cn(
                  "cursor-pointer transition-colors",
                  selectedDetection?.id === det.id
                    ? "bg-primary/10"
                    : "hover:bg-muted/50"
                )}
              >
                <td className="px-4 py-3 text-sm font-mono text-muted-foreground">
                  #{index + 1}
                </td>
                <td className="px-4 py-3">
                  <span
                    className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium text-white"
                    style={{ backgroundColor: getParticleColor(det.particleType) }}
                  >
                    {det.particleType}
                  </span>
                </td>
                <td className="px-4 py-3 text-sm text-foreground">
                  <span className="font-medium">{det.polymerType}</span>
                  <span className="text-muted-foreground ml-1 hidden sm:inline">
                    ({getPolymerFullName(det.polymerType).split(' ')[0]})
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden max-w-20">
                      <div
                        className="h-full bg-primary transition-all"
                        style={{ width: `${det.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-sm text-foreground font-medium">
                      {(det.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-sm text-foreground">
                  {(det.ldirMatchScore * 100).toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
