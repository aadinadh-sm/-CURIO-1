export interface ClassProbabilities {
  [className: string]: number;
}

export interface WindowPrediction {
  window_idx: number;
  window_start_seconds: number;
  window_end_seconds: number;
  predicted_condition: string;
  confidence: number;
  probability_normal: number;
  abnormality_score: number;
  class_probabilities: ClassProbabilities;
}

export interface DiagnosisSummary {
  condition: string;
  confidence: number;
  class_probabilities: ClassProbabilities;
  session_probability_normal: number;
  session_probability_cpu: number;
  session_probability_memory: number;
  session_probability_disk: number;
  per_window_predictions: WindowPrediction[];
}

export interface AbnormalitySummary {
  mean_score: number;
  max_score: number;
  final_window_score: number;
  abnormal_window_count: number;
  abnormal_window_ratio: number;
  threshold: number;
  session_abnormal: boolean;
  per_window_scores: number[];
}

export interface EvidenceItem {
  feature_name: string;
  display_name?: string;
  observed_value: number;
  normal_reference: number;
  normal_std?: number;
  raw_z?: number;
  directional_score: number;
  direction: string;
  evidence_strength: 'strong' | 'moderate' | 'weak' | 'contradictory';
  subsystem?: string;
  human_readable_statement?: string;
  statement?: string;
}

export interface EvidenceSummary {
  condition: string;
  confidence: number;
  abnormality_score: number;
  overall_interpretation: string;
  supporting_evidence: EvidenceItem[];
  contradictory_evidence: EvidenceItem[];
  neutral_features: EvidenceItem[];
  all_evaluated_features?: EvidenceItem[];
  evidence_score: number;
  evidence_consistency: number;
  representative_window_idx: number;
  representative_window_time?: string;
}

export interface OnsetEvent {
  feature_name: string;
  onset_window_idx: number;
  onset_time_seconds: number;
  peak_z_dir: number;
}

export interface DiscoverySummary {
  predicted_condition: string;
  discovery_status: 'CONFIRMED_PROGRESSION' | 'SUSTAINED_PRESSURE' | 'INSUFFICIENT_TEMPORAL_EVIDENCE' | 'OPERATING_EQUILIBRIUM';
  observed_sequence: string[];
  canonical_sequence: string[];
  onset_events: OnsetEvent[];
  kendall_tau: number | null;
  normalized_tau: number | null;
  discovery_coverage: number;
  missing_features?: string[];
  unexpected_features?: string[];
  interpretation: string;
}

export interface PerformanceMetrics {
  capture_duration_seconds: number;
  feature_extraction_ms: number;
  inference_ms: number;
  evidence_ms: number;
  discovery_ms: number;
  reporting_ms?: number;
  total_analysis_ms: number;
}

export interface DiagnosisResult {
  session_id: string;
  capture_started_at: string;
  capture_duration_seconds: number;
  raw_samples_count: number;
  feature_windows_count: number;
  diagnosis: DiagnosisSummary;
  abnormality: AbnormalitySummary;
  evidence: EvidenceSummary;
  discovery: DiscoverySummary;
  performance: PerformanceMetrics;
  metadata: {
    model_version: string;
    abnormality_threshold: number;
    feature_names: string[];
    pipeline_version: string;
  };
}

export interface DiagnosisStatusResponse {
  diagnosis_id: string;
  state: 'idle' | 'collecting' | 'analyzing' | 'complete' | 'cancelled' | 'error';
  stage: string;
  elapsed_seconds: number;
  progress: number;
  samples_collected: number;
  total_samples: number;
  started_at: string;
  result?: DiagnosisResult;
  error?: string;
}

export interface HistorySummaryItem {
  session_id: string;
  timestamp: string;
  condition: string;
  confidence: number;
  session_abnormal: boolean;
  abnormality_score: number;
  duration_seconds: number;
  discovery_status: string;
}

export interface SystemStatus {
  ready: boolean;
  model_loaded: boolean;
  deployment_dir: string;
  classes: string[];
  abnormality_threshold: number;
  features_count: number;
  history_count: number;
  local_only: boolean;
}
