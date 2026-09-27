import React from 'react';
import { X, Terminal } from 'lucide-react';
import type { DiagnosisResult } from '../types';

interface TechnicalModalProps {
  result: DiagnosisResult;
  onClose: () => void;
}

export const TechnicalModal: React.FC<TechnicalModalProps> = ({ result, onClose }) => {
  const diag = result.diagnosis;
  const abn = result.abnormality;
  const ev = result.evidence;
  const perf = result.performance;

  const evaluatedFeatures =
    ev.all_evaluated_features ||
    [...ev.supporting_evidence, ...ev.neutral_features, ...ev.contradictory_evidence];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.85rem' }}>
          <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#fff', letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Terminal size={18} style={{ color: 'var(--text-muted)' }} />
              <span>Technical Diagnostics & Telemetry Inspection</span>
            </h2>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: '0.15rem' }}>
              Session: {result.session_id} | Model: {result.metadata.model_version}
            </div>
          </div>
          <button
            onClick={onClose}
            style={{ color: 'var(--text-muted)', cursor: 'pointer', padding: '0.35rem', borderRadius: 'var(--radius-xs)', display: 'flex' }}
          >
            <X size={18} />
          </button>
        </div>

        {/* 1. Class Probabilities & Abnormality */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '1.75rem' }}>
          <div style={{ background: '#0d0d10', padding: '1.1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700, fontFamily: 'var(--font-mono)', marginBottom: '0.65rem' }}>
              Calibrated Class Probabilities (Mean across 11 windows)
            </h3>
            {Object.entries(diag.class_probabilities || {}).map(([cName, pVal]) => (
              <div key={cName} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
                <span style={{ color: cName === diag.condition ? '#fafafa' : 'var(--text-secondary)' }}>
                  {cName}:
                </span>
                <span style={{ fontWeight: 600, color: cName === diag.condition ? '#38bdf8' : 'var(--text-muted)' }}>
                  {(pVal * 100).toFixed(2)}% ({pVal.toFixed(4)})
                </span>
              </div>
            ))}
          </div>

          <div style={{ background: '#0d0d10', padding: '1.1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ fontSize: '0.74rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700, fontFamily: 'var(--font-mono)', marginBottom: '0.65rem' }}>
              Abnormality Assessment
            </h3>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Deployment Threshold:</span>
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{abn.threshold.toFixed(4)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Mean Abnormality Score:</span>
              <span style={{ fontWeight: 600, color: abn.session_abnormal ? '#fbbf24' : '#34d399' }}>{abn.mean_score.toFixed(4)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Max Abnormality Score:</span>
              <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{abn.max_score.toFixed(4)}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.35rem', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Abnormal Windows:</span>
              <span style={{ fontWeight: 600, color: abn.session_abnormal ? '#fbbf24' : '#34d399' }}>
                {abn.abnormal_window_count} / 11 ({(abn.abnormal_window_ratio * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
        </div>

        {/* 2. 11-Window Timeline Trajectory */}
        <div style={{ marginBottom: '1.75rem' }}>
          <h3 style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.65rem', color: '#fff', letterSpacing: '-0.01em' }}>
            11-Window Diagnostic Trajectory
          </h3>
          <div style={{ overflowX: 'auto', background: '#0d0d10', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
            <table className="curio-table" style={{ fontSize: '0.78rem' }}>
              <thead>
                <tr>
                  <th>Win</th>
                  <th>Interval</th>
                  <th>Condition</th>
                  <th>P(Normal)</th>
                  <th>Abnormality</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {diag.per_window_predictions &&
                  diag.per_window_predictions.map((w) => {
                    const isWinAbnormal = w.abnormality_score >= abn.threshold;
                    return (
                      <tr key={w.window_idx}>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{w.window_idx}</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>
                          {w.window_start_seconds.toFixed(1)}s - {w.window_end_seconds.toFixed(1)}s
                        </td>
                        <td style={{ fontWeight: 600, color: w.predicted_condition === 'normal' ? 'var(--text-secondary)' : '#fafafa' }}>
                          {w.predicted_condition}
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{(w.probability_normal * 100).toFixed(1)}%</td>
                        <td style={{ fontFamily: 'var(--font-mono)', color: isWinAbnormal ? '#fbbf24' : 'var(--text-muted)' }}>
                          {w.abnormality_score.toFixed(3)}
                        </td>
                        <td>
                          <span
                            style={{
                              padding: '0.15rem 0.45rem',
                              borderRadius: 'var(--radius-xs)',
                              fontSize: '0.65rem',
                              fontWeight: 700,
                              fontFamily: 'var(--font-mono)',
                              background: isWinAbnormal ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                              color: isWinAbnormal ? '#fbbf24' : '#34d399',
                              border: isWinAbnormal ? '1px solid rgba(245, 158, 11, 0.25)' : '1px solid rgba(16, 185, 129, 0.25)',
                            }}
                          >
                            {isWinAbnormal ? 'ABNORMAL' : 'OK'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        </div>

        {/* 3. Performance Latency Breakdown */}
        <div style={{ marginBottom: '1.75rem' }}>
          <h3 style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.65rem', color: '#fff', letterSpacing: '-0.01em' }}>
            Performance Latency Breakdown
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '0.65rem' }}>
            <div style={{ background: '#0d0d10', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '0.75rem' }}>
              <span className="metric-label">Capture Duration</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 600 }}>
                {perf.capture_duration_seconds.toFixed(1)}s
              </div>
            </div>

            <div style={{ background: '#0d0d10', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '0.75rem' }}>
              <span className="metric-label">Feature Extraction</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 600 }}>
                {perf.feature_extraction_ms.toFixed(1)} ms
              </div>
            </div>

            <div style={{ background: '#0d0d10', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '0.75rem' }}>
              <span className="metric-label">ML Inference</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 600, color: '#38bdf8' }}>
                {perf.inference_ms.toFixed(1)} ms
              </div>
            </div>

            <div style={{ background: '#0d0d10', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '0.75rem' }}>
              <span className="metric-label">Evidence Engine</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 600 }}>
                {perf.evidence_ms.toFixed(1)} ms
              </div>
            </div>

            <div style={{ background: '#0d0d10', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '0.75rem' }}>
              <span className="metric-label">Discovery Engine</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 600 }}>
                {perf.discovery_ms.toFixed(1)} ms
              </div>
            </div>

            <div style={{ background: '#0d0d10', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '0.75rem' }}>
              <span className="metric-label">Total Analysis</span>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.95rem', fontWeight: 700, color: '#34d399' }}>
                {perf.total_analysis_ms.toFixed(1)} ms
              </div>
            </div>
          </div>
        </div>

        {/* 4. Evaluated Feature Set Matrix */}
        <div>
          <h3 style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.65rem', color: '#fff', letterSpacing: '-0.01em' }}>
            Representative Window Feature Attribution Matrix
          </h3>
          <div style={{ overflowX: 'auto', background: '#0d0d10', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
            <table className="curio-table" style={{ fontSize: '0.78rem' }}>
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Observed</th>
                  <th>Normal Baseline</th>
                  <th>Directional Score</th>
                  <th>Strength</th>
                </tr>
              </thead>
              <tbody>
                {evaluatedFeatures.map((item, idx) => (
                  <tr key={idx}>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{item.feature_name}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>
                      {typeof item.observed_value === 'number' ? item.observed_value.toFixed(2) : String(item.observed_value)}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      {typeof item.normal_reference === 'number' ? item.normal_reference.toFixed(2) : '—'}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)', color: item.directional_score > 0 ? '#38bdf8' : 'var(--text-muted)' }}>
                      {typeof item.directional_score === 'number' ? item.directional_score.toFixed(2) : '0.00'}
                    </td>
                    <td>
                      <span className={`evidence-strength-tag ${item.evidence_strength}`}>
                        {item.evidence_strength}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
