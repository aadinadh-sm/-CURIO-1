import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { HomeScreen } from './components/HomeScreen';
import { DiagnosisScreen } from './components/DiagnosisScreen';
import { ResultScreen } from './components/ResultScreen';
import { HistoryScreen } from './components/HistoryScreen';
import { TechnicalModal } from './components/TechnicalModal';
import type { DiagnosisResult } from './types';

// Mock data
const mockNormalResult: DiagnosisResult = {
  session_id: 'test_normal_session',
  capture_started_at: '2026-09-28T00:00:00Z',
  capture_duration_seconds: 30.0,
  raw_samples_count: 61,
  feature_windows_count: 11,
  diagnosis: {
    condition: 'normal',
    confidence: 0.9543,
    class_probabilities: { normal: 0.9543, cpu_pressure: 0.015, memory_pressure: 0.015, disk_io_pressure: 0.0157 },
    session_probability_normal: 0.9543,
    session_probability_cpu: 0.015,
    session_probability_memory: 0.015,
    session_probability_disk: 0.0157,
    per_window_predictions: [
      { window_idx: 0, window_start_seconds: 0.0, window_end_seconds: 5.0, predicted_condition: 'normal', confidence: 0.95, probability_normal: 0.95, abnormality_score: 0.05, class_probabilities: { normal: 0.95 } }
    ],
  },
  abnormality: {
    mean_score: 0.045,
    max_score: 0.051,
    final_window_score: 0.048,
    abnormal_window_count: 0,
    abnormal_window_ratio: 0.0,
    threshold: 0.8536,
    session_abnormal: false,
    per_window_scores: [0.045],
  },
  evidence: {
    condition: 'normal',
    confidence: 0.9543,
    abnormality_score: 0.045,
    overall_interpretation: 'All observed telemetry signals remain consistent with the learned Normal operating reference.',
    supporting_evidence: [],
    contradictory_evidence: [],
    neutral_features: [],
    evidence_score: 0.0,
    evidence_consistency: 1.0,
    representative_window_idx: 0,
  },
  discovery: {
    predicted_condition: 'normal',
    discovery_status: 'OPERATING_EQUILIBRIUM',
    observed_sequence: [],
    canonical_sequence: [],
    onset_events: [],
    kendall_tau: null,
    normalized_tau: null,
    discovery_coverage: 0.0,
    interpretation: 'Operating Equilibrium: No escalation detected. Telemetry remained within learned Normal bounds.',
  },
  performance: {
    capture_duration_seconds: 30.0,
    feature_extraction_ms: 10.0,
    inference_ms: 120.0,
    evidence_ms: 2.5,
    discovery_ms: 1.5,
    total_analysis_ms: 134.0,
  },
  metadata: {
    model_version: 'calibrated_rf_sigmoid',
    abnormality_threshold: 0.8536,
    feature_names: ['cpu_mean', 'cpu_max'],
    pipeline_version: '1.0.0',
  },
};

const mockAbnormalResult: DiagnosisResult = {
  ...mockNormalResult,
  session_id: 'test_cpu_session',
  diagnosis: {
    ...mockNormalResult.diagnosis,
    condition: 'cpu_pressure',
    confidence: 0.961,
  },
  abnormality: {
    ...mockNormalResult.abnormality,
    mean_score: 0.988,
    abnormal_window_count: 11,
    abnormal_window_ratio: 1.0,
    session_abnormal: true,
  },
  evidence: {
    ...mockNormalResult.evidence,
    condition: 'cpu_pressure',
    supporting_evidence: [
      {
        feature_name: 'cpu_mean',
        display_name: 'Average CPU Utilization',
        observed_value: 100.0,
        normal_reference: 18.62,
        directional_score: 10.0,
        direction: 'elevated',
        evidence_strength: 'strong',
        human_readable_statement: 'Average CPU Utilization (100.0%) was substantially elevated.',
      },
    ],
  },
  discovery: {
    predicted_condition: 'cpu_pressure',
    discovery_status: 'SUSTAINED_PRESSURE',
    observed_sequence: ['cpu_max', 'cpu_mean'],
    canonical_sequence: ['cpu_max', 'cpu_mean'],
    onset_events: [
      { feature_name: 'cpu_max', onset_window_idx: 0, onset_time_seconds: 0.0, peak_z_dir: 5.79 },
      { feature_name: 'cpu_mean', onset_window_idx: 0, onset_time_seconds: 0.0, peak_z_dir: 10.0 },
    ],
    kendall_tau: null,
    normalized_tau: null,
    discovery_coverage: 1.0,
    interpretation: 'Sustained operating pressure detected; no transition observed during capture.',
  },
};

describe('CURIO Frontend Unit Tests', () => {
  it('1. Home screen renders headline and local-only privacy badge', () => {
    render(
      <HomeScreen
        onStartLiveDiagnosis={vi.fn()}
        onOpenHistory={vi.fn()}
        onOpenReplay={vi.fn()}
        onToggleTechnical={vi.fn()}
        technicalMode={false}
      />
    );
    expect(screen.getByText(/Understand what your/i)).toBeDefined();
    expect(screen.getByText(/Diagnose My Computer/i)).toBeDefined();
    expect(screen.getByText(/Local Only/i)).toBeDefined();
  });

  it('2. DiagnosisScreen shows real progress and telemetry subsystems', async () => {
    // Mock fetch for diagnosis polling
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        diagnosis_id: 'test_diag_123',
        state: 'collecting',
        stage: 'Collecting telemetry... (20/61 samples)',
        elapsed_seconds: 10.0,
        progress: 0.328,
        samples_collected: 20,
        total_samples: 61,
        started_at: '2026-09-28T00:00:00Z',
      }),
    }) as any;

    render(
      <DiagnosisScreen
        diagnosisId="test_diag_123"
        onComplete={vi.fn()}
        onCancel={vi.fn()}
        onError={vi.fn()}
      />
    );

    expect(screen.getByText(/CURIO is analyzing your computer/i)).toBeDefined();
    expect(screen.getByText(/CPU/i)).toBeDefined();
    expect(screen.getByText(/Memory/i)).toBeDefined();
    expect(screen.getByText(/Cancel Diagnosis/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText(/Collecting: 20 \/ 61 samples/i)).toBeDefined();
    });
  });

  it('3. ResultScreen renders Normal Operating Equilibrium cleanly', () => {
    render(
      <ResultScreen
        result={mockNormalResult}
        onNewDiagnosis={vi.fn()}
        onViewHistory={vi.fn()}
      />
    );

    expect(screen.getByText(/NORMAL OPERATING STATE/i)).toBeDefined();
    expect(screen.getByText(/Normal Operation/i)).toBeDefined();
    expect(screen.getByText(/95%/i)).toBeDefined();
    expect(screen.getAllByText(/Operating Equilibrium/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/No escalation detected/i).length).toBeGreaterThan(0);
  });

  it('4. ResultScreen renders Abnormal CPU Pressure with evidence cards and model confidence', () => {
    render(
      <ResultScreen
        result={mockAbnormalResult}
        onNewDiagnosis={vi.fn()}
        onViewHistory={vi.fn()}
      />
    );

    expect(screen.getByText(/ABNORMAL OPERATING STATE/i)).toBeDefined();
    expect(screen.getAllByText(/Cpu Pressure/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/96%/i)).toBeDefined();
    expect(screen.getByText(/Why CURIO Thinks This/i)).toBeDefined();
    expect(screen.getAllByText(/Average CPU Utilization/i).length).toBeGreaterThan(0);
  });

  it('5. ResultScreen renders Sustained Pressure fallback banner', () => {
    render(
      <ResultScreen
        result={mockAbnormalResult}
        onNewDiagnosis={vi.fn()}
        onViewHistory={vi.fn()}
      />
    );

    expect(screen.getAllByText(/Sustained Operating Pressure/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/no temporal transition was observed/i)).toBeDefined();
  });

  it('6. ResultScreen renders Insufficient Temporal Evidence fallback', () => {
    const insufficientResult: DiagnosisResult = {
      ...mockAbnormalResult,
      discovery: {
        ...mockAbnormalResult.discovery,
        discovery_status: 'INSUFFICIENT_TEMPORAL_EVIDENCE',
        interpretation: 'Not enough temporal evidence was available.',
      },
    };

    render(
      <ResultScreen
        result={insufficientResult}
        onNewDiagnosis={vi.fn()}
        onViewHistory={vi.fn()}
      />
    );

    expect(screen.getByText(/Insufficient Temporal Evidence/i)).toBeDefined();
  });

  it('7. HistoryScreen displays records loaded from API', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [
        {
          session_id: 'sess_hist_01',
          timestamp: '2026-09-28T00:15:00Z',
          condition: 'memory_pressure',
          confidence: 0.91,
          session_abnormal: true,
          abnormality_score: 0.94,
          duration_seconds: 30.0,
          discovery_status: 'SUSTAINED_PRESSURE',
        },
      ],
    }) as any;

    render(
      <HistoryScreen
        onSelectRecord={vi.fn()}
        onBackToHome={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Memory Pressure/i)).toBeDefined();
      expect(screen.getByText(/ABNORMAL/i)).toBeDefined();
      expect(screen.getByText(/91%/i)).toBeDefined();
    });
  });

  it('8. TechnicalModal renders class probabilities and performance timings', () => {
    render(
      <TechnicalModal
        result={mockAbnormalResult}
        onClose={vi.fn()}
      />
    );

    expect(screen.getByText(/Technical Diagnostics & Telemetry Inspection/i)).toBeDefined();
    expect(screen.getByText(/Deployment Threshold:/i)).toBeDefined();
    expect(screen.getByText(/Performance Latency Breakdown/i)).toBeDefined();
    expect(screen.getByText(/120.0 ms/i)).toBeDefined();
  });

  it('9. Diagnosis cancellation shows restored system state', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        diagnosis_id: 'test_cancel_id',
        state: 'cancelled',
        stage: 'Diagnosis cancelled. System state restored.',
        elapsed_seconds: 5.0,
        progress: 0.16,
        samples_collected: 10,
        total_samples: 61,
        started_at: '2026-09-28T00:00:00Z',
      }),
    }) as any;

    render(
      <DiagnosisScreen
        diagnosisId="test_cancel_id"
        onComplete={vi.fn()}
        onCancel={vi.fn()}
        onError={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getAllByText(/Diagnosis Cancelled/i).length).toBeGreaterThan(0);
      expect(screen.getByText(/System state restored/i)).toBeDefined();
    });
  });

  it('10. Diagnosis error displays friendly diagnostic failure message', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        diagnosis_id: 'test_err_id',
        state: 'error',
        stage: 'Error encountered during diagnosis.',
        error: 'Hardware sensor communication failure.',
        elapsed_seconds: 5.0,
        progress: 0.16,
        samples_collected: 10,
        total_samples: 61,
        started_at: '2026-09-28T00:00:00Z',
      }),
    }) as any;

    render(
      <DiagnosisScreen
        diagnosisId="test_err_id"
        onComplete={vi.fn()}
        onCancel={vi.fn()}
        onError={vi.fn()}
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/Diagnostic Error/i)).toBeDefined();
      expect(screen.getByText(/Hardware sensor communication failure/i)).toBeDefined();
    });
  });
});
