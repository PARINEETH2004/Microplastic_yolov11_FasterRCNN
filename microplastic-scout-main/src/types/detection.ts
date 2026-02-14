export type ParticleType = 'fiber' | 'fragment' | 'film' | 'pellet' | 'foam';

export type PolymerType = 'PE' | 'PP' | 'PS' | 'PET' | 'PVC' | 'PA' | 'Unknown';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Detection {
  id: string;
  particleType: ParticleType;
  polymerType: PolymerType;
  confidence: number;
  boundingBox: BoundingBox;
  ldirMatchScore: number;
  spectrumData: number[];
  algorithm?: 'yolo' | 'faster_rcnn';
}

export interface DetectionResult {
  imageUrl: string;
  imageName: string;
  timestamp: Date;
  mode: 'fast' | 'accurate';
  processingTime: number;
  detections: Detection[];
  totalCount: number;
  countByType: Record<ParticleType, number>;
}

export type DetectionMode = 'fast' | 'accurate';
