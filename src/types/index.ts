export interface AnalysisResult {
  isFake: boolean;
  confidence: number;
  processingTime: number;
  frameAnalysis: {
    frameNumber: number;
    confidence: number;
    isFake: boolean;
  }[];
  metadata: {
    videoDuration: number;
    frameCount: number;
    resolution: string;
    format: string;
  };
}

export interface FrameAnalysis {
  frameNumber: number;
  timestamp: number;
  confidence: number;
  pathAScore: number;
  pathBScore: number;
  fusionScore: number;
  asciiRepresentation: string;
  detectedArtifacts: string[];
}

export interface TemporalAnomaly {
  startFrame: number;
  endFrame: number;
  anomalType: 'compression' | 'blending' | 'temporal' | 'frequency';
  severity: number;
  description: string;
}

export interface ModelMetrics {
  inceptionScore: number;
  efficientNetScore: number;
  lstmScore: number;
  beadalFeatures: number;
  computeReduction: number;
}

export interface TamperRegion {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  frameNumber: number;
}

export interface SystemStatus {
  gpuUtilization: number;
  queueLength: number;
  processedVideos: number;
  averageLatency: number;
  modelAccuracy: number;
  systemHealth: 'healthy' | 'warning' | 'critical';
}