export interface AnalysisResult {
  verdict: "REAL" | "FAKE" | "UNCERTAIN";
  confidence: number;
  prob: number;
  logit: number;
  spatial_weight: number;
  freq_weight: number;
  gradcam_base64: string;
  faces_detected: number;
  faces_total: number;
  analysis_time: number;
  ai_explanation: string;
}

export interface AnalysisError {
  message: string;
  detail?: string;
}