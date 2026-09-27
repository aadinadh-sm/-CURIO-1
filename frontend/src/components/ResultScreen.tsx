import React, { useState } from 'react';
import { ArrowLeft, Clock, Activity, AlertTriangle, CheckCircle2, Info, Terminal } from 'lucide-react';
import type { DiagnosisResult } from '../types';
import { TechnicalModal } from './TechnicalModal';

interface ResultScreenProps {
  result: DiagnosisResult;
  onNewDiagnosis: () => void;
  onViewHistory: () => void;
  defaultTechnicalOpen?: boolean;
}

export const ResultScreen: React.FC<ResultScreenProps> = ({
  result,
  onNewDiagnosis,
  onViewHistory,
  defaultTechnicalOpen = false,
}) => {
  const [showTechnical, setShowTechnical] = useState(defaultTechnicalOpen);

  const diag = result.diagnosis;
  const abn = result.abnormality;
  const ev = result.evidence;
  const disc = result.discovery;

  const isNormal = diag.condition === 'normal';
  const sessionAbnormal = abn.session_abnormal;
  const confidencePct = Math.round(diag.confidence * 100);
  const conditionDisplay = isNormal
    ? 'Normal Operation'
    : diag.condition.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  const discStatus = disc.discovery_status;
  const hasValidTau = disc.normalized_tau !== null && disc.kendall_tau !== null;

  return (
    <div>
      {/* Top Action Toolbar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
        <button
          onClick={onNewDiagnosis}
          style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', color: 'var(--text-secondary)', fontSize: '0.85rem', fontWeight: 500 }}
        >
          <ArrowLeft size={14} />
          <span>Run New Diagnosis</span>
        </button>

        <div style={{ display: 'flex', gap: '0.65rem' }}>
          <button
            className="btn-secondary"
            onClick={() => setShowTechnical(!showTechnical)}
            style={{ fontSize: '0.82rem', padding: '0.45rem 0.85rem' }}
          >
            <Terminal size={14} />
            <span>{showTechnical ? 'Close Technical Details' : 'View Technical Details'}</span>
          </button>
          <button
            className="btn-secondary"
            onClick={onViewHistory}
            style={{ fontSize: '0.82rem', padding: '0.45rem 0.85rem' }}
          >
            <Clock size={14} />
            <span>History List</span>
          </button>
        </div>
      </div>

      {/* Main Diagnosis Summary Card */}
      <div className={`result-header-card ${sessionAbnormal ? 'abnormal' : 'normal'}`}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1.5rem' }}>
          <div style={{ flex: 1, minWidth: '300px' }}>
            {result.input_source === 'uploaded_csv' && (
              <div className="result-source-label"><Activity size={12} /> CSV analysis · {result.input_filename || 'Uploaded capture'}</div>
            )}
            <div className={`status-pill ${sessionAbnormal ? 'abnormal' : 'normal'}`}>
              {sessionAbnormal ? <AlertTriangle size={12} /> : <CheckCircle2 size={12} />}
              <span>{sessionAbnormal ? 'ABNORMAL OPERATING STATE' : 'NORMAL OPERATING STATE'}</span>
            </div>
            <h1 className="condition-title">{conditionDisplay}</h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', maxWidth: '600px', lineHeight: 1.5 }}>
              {isNormal
                ? 'All operating signals remained well within the learned Normal equilibrium boundaries throughout the 30-second capture.'
                : `System exhibited patterns consistent with ${conditionDisplay.toLowerCase()} across multiple telemetry dimensions.`}
            </p>
          </div>

          <div className="metrics-badge-row">
            <div className="metric-item">
              <span className="metric-label">Model Confidence</span>
              <span className="metric-value">{confidencePct}%</span>
            </div>

            <div className="metric-item" style={{ borderLeft: '1px solid var(--border-subtle)', paddingLeft: '1.25rem' }}>
              <span className="metric-label">Abnormality Score</span>
              <span className="metric-value" style={{ color: sessionAbnormal ? 'var(--status-abnormal)' : 'var(--status-normal)' }}>
                {abn.mean_score.toFixed(2)}
              </span>
            </div>

            <div className="metric-item" style={{ borderLeft: '1px solid var(--border-subtle)', paddingLeft: '1.25rem' }}>
              <span className="metric-label">Abnormal Windows</span>
              <span className="metric-value">{abn.abnormal_window_count} / 11</span>
            </div>
          </div>
        </div>
      </div>

      {/* SECTION: WHY CURIO THINKS THIS (Evidence Attribution) */}
      <div className="section-block">
        <h2 className="section-heading">
          <Info size={16} style={{ color: 'var(--accent-zinc)' }} />
          <span>{isNormal ? 'Operating Reference Alignment' : 'Why CURIO Thinks This'}</span>
        </h2>

        {isNormal ? (
          <div className="discovery-card" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.4rem' }}>
              <CheckCircle2 size={16} style={{ color: 'var(--status-normal)' }} />
              <span>Telemetry remained consistent with the learned Normal operating reference.</span>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.6, paddingLeft: '1.5rem' }}>
              None of the 12 extracted telemetry features crossed the statistical abnormality threshold ({abn.threshold.toFixed(2)}) for the required two-window confirmation period.
            </p>
          </div>
        ) : (
          <div className="evidence-cards-grid">
            {ev.supporting_evidence.map((item, idx) => (
              <div key={idx} className="evidence-card">
                <div className="evidence-feature-header">
                  <div>
                    <div className="evidence-feature-name">{item.display_name || item.feature_name}</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {item.feature_name}
                    </div>
                  </div>
                  <span className={`evidence-strength-tag ${item.evidence_strength}`}>
                    {item.evidence_strength}
                  </span>
                </div>

                <div className="evidence-values-row">
                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', display: 'block' }}>Observed</span>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {item.feature_name.includes('pct') ? `${item.observed_value.toFixed(1)}%` : item.observed_value.toFixed(2)}
                    </span>
                  </div>

                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', display: 'block' }}>Normal Reference</span>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {item.feature_name.includes('pct') ? `${item.normal_reference.toFixed(1)}%` : item.normal_reference.toFixed(2)}
                    </span>
                  </div>

                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', display: 'block' }}>Direction</span>
                    <span style={{ color: item.direction === 'elevated' || item.direction === 'increased' ? 'var(--status-abnormal)' : '#38bdf8' }}>
                      {item.direction === 'elevated' || item.direction === 'increased' ? '↑ Elevated' : item.direction === 'reduced' || item.direction === 'decreased' ? '↓ Reduced' : '— Stable'}
                    </span>
                  </div>
                </div>

                <p className="evidence-statement-text">
                  {item.human_readable_statement || item.statement}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* SECTION: CURIO DISCOVERY (Temporal Engine) */}
      <div className="section-block">
        <h2 className="section-heading">
          <Activity size={16} style={{ color: 'var(--accent-zinc)' }} />
          <span>CURIO Discovery</span>
        </h2>

        <div className="discovery-card">
          {discStatus === 'OPERATING_EQUILIBRIUM' && (
            <div className="discovery-fallback-banner">
              <strong style={{ color: 'var(--status-normal)', display: 'block', marginBottom: '0.25rem' }}>
                Operating Equilibrium
              </strong>
              No escalation detected. Telemetry remained within learned Normal bounds throughout the 30-second capture.
            </div>
          )}

          {discStatus === 'SUSTAINED_PRESSURE' && (
            <div className="discovery-fallback-banner" style={{ borderLeft: '3px solid var(--status-abnormal)' }}>
              <strong style={{ color: 'var(--status-abnormal)', display: 'block', marginBottom: '0.25rem' }}>
                Sustained Operating Pressure
              </strong>
              Sustained operating pressure detected. No temporal transition was observed during this capture because the workload was already active at start ($t=0.0$s).
            </div>
          )}

          {discStatus === 'INSUFFICIENT_TEMPORAL_EVIDENCE' && (
            <div className="discovery-fallback-banner">
              <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.25rem' }}>
                Insufficient Temporal Evidence
              </strong>
              Not enough temporal evidence was available to establish a reliable progression pattern (fewer than 3 distinct non-tied onsets).
            </div>
          )}

          {/* Onset Timeline if observed sequence exists */}
          {disc.observed_sequence && disc.observed_sequence.length > 0 && (
            <div style={{ marginTop: '1.25rem' }}>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginBottom: '0.65rem' }}>
                Observed Onset Timeline (30.0s Window)
              </div>

              <div className="timeline-track-list">
                {disc.observed_sequence.map((featName, idx) => {
                  const evInfo = disc.onset_events?.find((e) => e.feature_name === featName);
                  const onsetTime = evInfo ? evInfo.onset_time_seconds : 0.0;
                  const positionPct = Math.min(95, Math.max(5, (onsetTime / 30.0) * 100));

                  return (
                    <div key={idx} className="timeline-row">
                      <div className="timeline-feature-label" title={featName}>
                        {featName}
                      </div>

                      <div className="timeline-bar-container">
                        <div
                          className="timeline-onset-marker"
                          style={{ left: `${positionPct}%` }}
                          title={`Onset confirmed at t = ${onsetTime.toFixed(1)}s`}
                        ></div>
                      </div>

                      <div className="timeline-onset-time">
                        t = {onsetTime.toFixed(1)}s
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Similarity & Coverage Row */}
          <div style={{ display: 'flex', gap: '2rem', marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-subtle)', flexWrap: 'wrap' }}>
            <div>
              <span className="metric-label">Temporal Similarity</span>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: hasValidTau ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                {hasValidTau ? `τ = ${disc.kendall_tau?.toFixed(2)} (normalized: ${disc.normalized_tau?.toFixed(2)})` : 'N/A'}
              </div>
            </div>

            <div>
              <span className="metric-label">Discovery Coverage</span>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                {Math.round(disc.discovery_coverage * 100)}%
              </div>
            </div>

            <div style={{ flex: 1, minWidth: '220px' }}>
              <span className="metric-label">Interpretation</span>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontStyle: 'italic', marginTop: '0.15rem' }}>
                "{disc.interpretation || ev.overall_interpretation}"
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Technical Diagnostics Modal */}
      {showTechnical && (
        <TechnicalModal
          result={result}
          onClose={() => setShowTechnical(false)}
        />
      )}
    </div>
  );
};
