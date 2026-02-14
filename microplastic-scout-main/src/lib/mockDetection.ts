import type { Detection, DetectionResult, ParticleType, PolymerType, DetectionMode } from '@/types/detection';

const particleTypes: ParticleType[] = ['fiber', 'fragment', 'film', 'pellet', 'foam'];
const polymerTypes: PolymerType[] = ['PE', 'PP', 'PS', 'PET', 'PVC', 'PA'];

function generateSpectrumData(): number[] {
  const data: number[] = [];
  for (let i = 0; i < 100; i++) {
    // Simulate IR spectrum with peaks
    const baseNoise = Math.random() * 0.1;
    const peak1 = Math.exp(-Math.pow((i - 25) / 8, 2)) * 0.7;
    const peak2 = Math.exp(-Math.pow((i - 55) / 12, 2)) * 0.9;
    const peak3 = Math.exp(-Math.pow((i - 80) / 6, 2)) * 0.5;
    data.push(baseNoise + peak1 + peak2 + peak3);
  }
  return data;
}

function generateRandomDetection(imageWidth: number, imageHeight: number): Detection {
  const particleType = particleTypes[Math.floor(Math.random() * particleTypes.length)];
  const polymerType = polymerTypes[Math.floor(Math.random() * polymerTypes.length)];
  
  const width = 30 + Math.random() * 80;
  const height = 20 + Math.random() * 60;
  
  return {
    id: `det-${Math.random().toString(36).substr(2, 9)}`,
    particleType,
    polymerType,
    confidence: 0.7 + Math.random() * 0.25,
    boundingBox: {
      x: Math.random() * (imageWidth - width),
      y: Math.random() * (imageHeight - height),
      width,
      height,
    },
    ldirMatchScore: 0.75 + Math.random() * 0.2,
    spectrumData: generateSpectrumData(),
  };
}

export async function simulateDetection(
  imageUrl: string,
  imageName: string,
  mode: DetectionMode
): Promise<DetectionResult> {
  // Simulate processing time
  const processingTime = mode === 'fast' ? 1500 + Math.random() * 1000 : 3000 + Math.random() * 2000;
  
  await new Promise(resolve => setTimeout(resolve, processingTime));
  
  // Generate random detections
  const numDetections = mode === 'fast' 
    ? 5 + Math.floor(Math.random() * 10)
    : 8 + Math.floor(Math.random() * 15);
  
  const detections: Detection[] = [];
  for (let i = 0; i < numDetections; i++) {
    detections.push(generateRandomDetection(640, 480));
  }
  
  // Calculate counts by type
  const countByType: Record<ParticleType, number> = {
    fiber: 0,
    fragment: 0,
    film: 0,
    pellet: 0,
    foam: 0,
  };
  
  detections.forEach(det => {
    countByType[det.particleType]++;
  });
  
  return {
    imageUrl,
    imageName,
    timestamp: new Date(),
    mode,
    processingTime,
    detections,
    totalCount: detections.length,
    countByType,
  };
}

export function getParticleColor(type: ParticleType): string {
  const colors: Record<ParticleType, string> = {
    fiber: 'hsl(200, 80%, 50%)',
    fragment: 'hsl(340, 75%, 55%)',
    film: 'hsl(45, 90%, 50%)',
    pellet: 'hsl(280, 60%, 55%)',
    foam: 'hsl(120, 50%, 45%)',
  };
  return colors[type];
}

export function getPolymerFullName(type: PolymerType): string {
  const names: Record<PolymerType, string> = {
    PE: 'Polyethylene',
    PP: 'Polypropylene',
    PS: 'Polystyrene',
    PET: 'Polyethylene Terephthalate',
    PVC: 'Polyvinyl Chloride',
    PA: 'Polyamide (Nylon)',
    Unknown: 'Unknown Polymer',
  };
  return names[type];
}
