import type { DetectionResult, DetectionMode } from '../types/detection';

const API_BASE_URL = '/api'; // Use proxy path

interface ApiResponse {
    imageUrl: string;
    imageName: string;
    timestamp: number;
    mode: DetectionMode;
    processingTime: number;
    detections: any[];
    totalCount: number;
    countByType: Record<string, number>;
    imageSize: {
        width: number;
        height: number;
    };
}

interface ApiError {
    error: string;
}

class YoloDetectionService {
    private async makeRequest<T>(url: string, options: RequestInit = {}): Promise<T> {
        // CRITICAL FIX: Don't set Content-Type for FormData requests
        const isFormData = options.body instanceof FormData;
        const headers = {
            // Only set Content-Type to application/json if NOT sending FormData
            ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
            ...options.headers,
        };

        const response = await fetch(url, {
            ...options,
            headers,
        });

        if (!response.ok) {
            const errorData: ApiError = await response.json().catch(() => ({ error: `HTTP error! status: ${response.status}` }));
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        return response.json();
    }

    async healthCheck(): Promise<{ status: string; model_loaded: boolean; version: string }> {
        return this.makeRequest(`${API_BASE_URL}/health`);
    }

    async getApiConfig(): Promise<any> {
        return this.makeRequest(`${API_BASE_URL}/config`);
    }

    async detectMicroplastics(
        imageFile: File,
        mode: DetectionMode = 'fast',
        algorithm: 'yolo' | 'faster_rcnn' = 'yolo'
    ): Promise<DetectionResult> {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('mode', mode);
        formData.append('algorithm', algorithm);

        try {
            const response: ApiResponse = await this.makeRequest(`${API_BASE_URL}/detect`, {
                method: 'POST',
                body: formData,
                headers: {
                    // Don't set Content-Type when using FormData
                },
            });

            // Convert API response to DetectionResult format
            const result: DetectionResult = {
                imageUrl: response.imageUrl,
                imageName: response.imageName,
                timestamp: new Date(response.timestamp * 1000),
                mode: response.mode,
                processingTime: response.processingTime,
                detections: response.detections,
                totalCount: response.totalCount,
                countByType: response.countByType as Record<any, number>,
            };

            return result;
        } catch (error) {
            throw error;
        }
    }

    async isBackendAvailable(): Promise<boolean> {
        try {
            await this.healthCheck();
            return true;
        } catch (error) {
            console.warn('Backend not available, falling back to mock detection:');
            return false;
        }
    }
}

// Create singleton instance
export const yoloDetectionService = new YoloDetectionService();

// Export for use in components
export async function detectWithYolo(
    imageFile: File,
    mode: DetectionMode = 'fast',
    algorithm: 'yolo' | 'faster_rcnn' = 'yolo'
): Promise<DetectionResult> {
    try {
        const isAvailable = await yoloDetectionService.isBackendAvailable();

        if (isAvailable) {
            const result = await yoloDetectionService.detectMicroplastics(imageFile, mode, algorithm);
            return result;
        } else {
            const { simulateDetection } = await import('./mockDetection');
            const mockResult = simulateDetection(URL.createObjectURL(imageFile), imageFile.name, mode);
            return mockResult;
        }
    } catch (error) {
        // Fallback to mock detection on any error
        // Always fallback to mock detection on any error
        try {
            const { simulateDetection } = await import('./mockDetection');
            const mockResult = await simulateDetection(URL.createObjectURL(imageFile), imageFile.name, mode);
            return mockResult;
        } catch (mockError) {
            throw new Error('Both backend and mock detection failed');
        }
    }
}

// Add detailed debugging function that can be called from components
export function debugFileObject(file: File): string[] {
    return [
        `File Object Debug Info:`,
        `Name: ${file.name}`,
        `Size: ${file.size} bytes`,
        `Type: ${file.type}`,
        `Last Modified: ${new Date(file.lastModified).toLocaleString()}`,
        `Instance of File: ${file instanceof File}`,
        `Has Blob methods: ${typeof file.slice === 'function'}`,
        `File constructor name: ${file.constructor.name}`
    ];
}
