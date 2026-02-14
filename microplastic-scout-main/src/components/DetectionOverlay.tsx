import { useState, useEffect } from 'react';
import type { Detection, DetectionResult } from '@/types/detection';
import { getParticleColor } from '@/lib/mockDetection';
import { cn } from '@/lib/utils';

interface DetectionOverlayProps {
  result: DetectionResult;
  selectedDetection: Detection | null;
  onSelectDetection: (detection: Detection | null) => void;
  originalImage?: string; // Add prop for original image URL
}

export function DetectionOverlay({
  result,
  selectedDetection,
  onSelectDetection,
  originalImage
}: DetectionOverlayProps) {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [imageSrc, setImageSrc] = useState<string>('');

  useEffect(() => {
    // Use original image if available, otherwise try to load from backend
    if (originalImage) {
      setImageSrc(originalImage);
    } else {
      // Fallback to the backend URL, but handle the case where it doesn't serve images
      setImageSrc(result.imageUrl);
    }
  }, [originalImage, result.imageUrl]);

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
  };

  const handleImageError = (e: React.SyntheticEvent<HTMLImageElement>) => {
    console.error('Failed to load image:', e.currentTarget.src);
    // If backend image fails, try to use a fallback or show error
    if (originalImage && e.currentTarget.src !== originalImage) {
      setImageSrc(originalImage);
    }
  };

  return (
    <div className="relative bg-muted rounded-xl overflow-hidden">
      <img
        src={imageSrc}
        alt="Analysis result"
        className="w-full h-auto"
        onLoad={handleImageLoad}
        onError={handleImageError}
      />

      {/* Bounding box overlays */}
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox={`0 0 ${imageSize.width || 640} ${imageSize.height || 480}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {result.detections.map((det) => (
          <g key={det.id}>
            {/* Bounding box */}
            <rect
              x={det.boundingBox.x}
              y={det.boundingBox.y}
              width={det.boundingBox.width}
              height={det.boundingBox.height}
              fill="transparent"
              stroke={getParticleColor(det.particleType)}
              strokeWidth={selectedDetection?.id === det.id ? 3 : 2}
              className={cn(
                "cursor-pointer transition-all",
                selectedDetection?.id === det.id && "animate-pulse"
              )}
              onClick={() => onSelectDetection(det)}
            />

            {/* Label background */}
            <rect
              x={det.boundingBox.x}
              y={det.boundingBox.y - 20}
              width={det.boundingBox.width}
              height={18}
              fill={getParticleColor(det.particleType)}
              rx={2}
            />

            {/* Label text */}
            <text
              x={det.boundingBox.x + 4}
              y={det.boundingBox.y - 6}
              fill="white"
              fontSize="10"
              fontFamily="Inter, sans-serif"
              fontWeight="500"
            >
              {det.particleType} {(det.confidence * 100).toFixed(0)}%
            </text>
          </g>
        ))}
      </svg>

      {/* Click hint */}
      <div className="absolute bottom-4 left-4 px-3 py-1.5 rounded-full glass-panel text-xs text-muted-foreground">
        Click on a detection to view LDIR spectrum
      </div>
    </div>
  );
}
