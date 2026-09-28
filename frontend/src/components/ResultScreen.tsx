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
  const conditionDisplay = isNormal ? 'No problem found' : ({
    cpu_pressure: 'CPU under heavy load',
    memory_pressure: 'Memory running low',
    disk_io_pressure: 'Disk under heavy load',
  }[diag.condition] || diag.condition.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()));
  const friendlySignalNames: Record<string, string> = {
    cpu_mean: 'Average processor use',
    cpu_max: 'Peak processor use',
    cpu_std: 'Changes in processor use',
    cpu_core_imbalance: 'How evenly processor work is shared',
    ram_used_pct: 'Memory in use',
    ram_available_ratio: 'Memory still available',
    swap_used_pct: 'Disk space used as extra memory',
    disk_io_rate_norm: 'Data read or written',
    disk_iops_norm: 'Disk activity rate',
    process_count_delta: 'Change in running apps',
    top_proc_cpu_ratio: 'Processor share used by the busiest app',
    top_proc_mem_pct: 'Memory used by the largest app',
  };
  const signalName = (name: string) => friendlySignalNames[name] || name.replace(/_/g, ' ');
  const formatSignalValue = (name: string, value: number) => {
    if (['cpu_mean', 'cpu_max', 'cpu_std', 'cpu_core_imbalance', 'ram_used_pct', 'swap_used_pct', 'top_proc_mem_pct'].includes(name)) return `${value.toFixed(1)}%`;
    if (['ram_available_ratio', 'top_proc_cpu_ratio'].includes(name)) return `${(value * 100).toFixed(1)}%`;
    if (name === 'process_count_delta') return `${value > 0 ? '+' : ''}${Math.round(value)} apps`;
    if (name === 'disk_io_rate_norm') {
      const bytesPerSecond = Math.max(0, (10 ** value) - 1);
      return bytesPerSecond >= 1_000_000 ? `${(bytesPerSecond / 1_000_000).toFixed(1)} MB/s` : `${(bytesPerSecond / 1_000).toFixed(0)} KB/s`;
    }
    if (name === 'disk_iops_norm') return `${Math.max(0, (10 ** value) - 1).toFixed(0)} disk actions/s`;
    return value.toFixed(2);
  };
  const resultSummary = ({
    cpu_pressure: 'CURIO found signs that your processor may be working too hard.',
    memory_pressure: 'CURIO found signs that your computer may be running low on memory.',
    disk_io_pressure: 'CURIO found signs that your disk may be very busy.',
  } as Record<string, string>)[diag.condition] || `CURIO found signs of ${conditionDisplay.toLowerCase()}.`;
  const generalNextStep = ({
    cpu_pressure: 'Open Task Manager, sort the Processes list by CPU, and see which app stays near the top while the slowdown is happening.',
    memory_pressure: 'Open Task Manager, sort the Processes list by Memory, and see whether one app keeps using more memory. Run CURIO again while the slowdown is happening.',
    disk_io_pressure: 'Open Task Manager, sort the Processes list by Disk, and see whether one app keeps the disk busy.',
  } as Record<string, string>)[diag.condition] || 'If your computer feels slow again, run CURIO while it is happening so the check can capture it.';
  const causeLimit = ({
    cpu_pressure: 'This check measures processor load, but it does not save app names. It cannot tell you which app used the processor.',
    memory_pressure: 'This check measures memory use, but it does not save app names. It cannot tell you which app used the memory or why.',
    disk_io_pressure: 'This check measures disk activity, but it does not save app names or file details. It cannot tell you what was using the disk.',
  } as Record<string, string>)[diag.condition] || 'This short check can show readings during the moment, but it cannot explain a slowdown it did not capture.';
  const processResource = diag.condition === 'cpu_pressure' ? 'processor' : 'memory';
  const rawProcessLeader = diag.condition === 'cpu_pressure'
    ? result.process_context?.cpu_leader
    : diag.condition === 'memory_pressure'
      ? result.process_context?.memory_leader
      : null;
  const processLeader = !isNormal && rawProcessLeader && rawProcessLeader.share_of_named_snapshots >= 0.2
    ? rawProcessLeader
    : null;
  const processPeak = processLeader
    ? diag.condition === 'cpu_pressure'
      ? `${processLeader.peak_value.toFixed(1)}% of total processor capacity`
      : `${(processLeader.peak_value / (1024 * 1024)).toFixed(0)} MB in use`
    : null;
  const nextStep = processLeader
    ? `In Task Manager, find ${processLeader.name} in Processes and check whether its ${processResource} use stays high while the slowdown continues.`
    : generalNextStep;
  const clueCount = ev.supporting_evidence.length;
  const conflictingCount = ev.contradictory_evidence.length;
  const evidenceSummary = isNormal
    ? 'The readings were closer to CURIO’s normal examples than its problem examples.'
    : clueCount === 0
      ? 'The model found a closest match, but the measured readings did not provide clear supporting clues.'
      : `CURIO found ${clueCount} ${clueCount === 1 ? 'reading that fits' : 'readings that fit'} this pattern${conflictingCount ? `, along with ${conflictingCount} ${conflictingCount === 1 ? 'reading that points another way' : 'readings that point another way'}` : ''}.`;
  const timingSummary = isNormal
    ? 'No clear problem pattern appeared during this check.'
    : disc.discovery_status === 'CONFIRMED_PROGRESSION' && disc.observed_sequence.length > 1
      ? `CURIO saw readings change in this order: ${disc.observed_sequence.slice(0, 3).map(signalName).join(' → ')}. Timing can suggest a pattern, but it does not prove that one change caused another.`
      : disc.discovery_status === 'SUSTAINED_PRESSURE'
        ? 'The signs were already present when the check began, so CURIO could not see what started first.'
        : 'There were not enough separate changes to tell which reading changed first.';

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
          <span>Run another check</span>
        </button>

        <div style={{ display: 'flex', gap: '0.65rem' }}>
          <button
            className="btn-secondary"
            onClick={() => setShowTechnical(!showTechnical)}
            style={{ fontSize: '0.82rem', padding: '0.45rem 0.85rem' }}
          >
            <Terminal size={14} />
            <span>{showTechnical ? 'Close technical details' : 'Technical details'}</span>
          </button>
          <button
            className="btn-secondary"
            onClick={onViewHistory}
            style={{ fontSize: '0.82rem', padding: '0.45rem 0.85rem' }}
          >
            <Clock size={14} />
            <span>Past checks</span>
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
              <span>{sessionAbnormal ? 'POSSIBLE ISSUE FOUND' : 'NO ISSUE FOUND'}</span>
            </div>
            <h1 className="condition-title">{conditionDisplay}</h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.92rem', maxWidth: '600px', lineHeight: 1.5 }}>
              {isNormal
                ? 'CURIO did not find signs of a problem during this check.'
                : `${resultSummary} This is the closest match to the readings CURIO checked.`}
            </p>
          </div>

          <div className="metrics-badge-row">
            <div className="metric-item">
              <span className="metric-label" title="How strongly CURIO favors this result. It is an estimate, not a guarantee.">How sure CURIO is</span>
              <span className="metric-value">{confidencePct}%</span>
            </div>

            <div className="metric-item" style={{ borderLeft: '1px solid var(--border-subtle)', paddingLeft: '1.25rem' }}>
              <span className="metric-label" title="A score showing how unusual the readings were. Higher means farther from the usual range.">Unusual reading score</span>
              <span className="metric-value" style={{ color: sessionAbnormal ? 'var(--status-abnormal)' : 'var(--status-normal)' }}>
                {abn.mean_score.toFixed(2)}
              </span>
            </div>

            <div className="metric-item" style={{ borderLeft: '1px solid var(--border-subtle)', paddingLeft: '1.25rem' }}>
              <span className="metric-label" title="CURIO splits the check into short periods and counts how many showed unusual readings.">Short periods flagged</span>
              <span className="metric-value">{abn.abnormal_window_count} of 11</span>
            </div>
          </div>
        </div>
      </div>

      {/* Likely cause and practical confirmation steps */}
      <section className="section-block">
        <h2 className="section-heading">
          <Info size={16} style={{ color: 'var(--accent-zinc)' }} />
          <span>Most likely area to check</span>
        </h2>
        <div className="discovery-card" style={{ padding: '1.25rem' }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.25rem' }}>
            <div>
              <div className="metric-label" style={{ marginBottom: '0.35rem' }}>{isNormal ? 'What this check found' : 'Likely issue'}</div>
              <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.45rem' }}>
                {isNormal ? 'No clear problem found' : conditionDisplay}
              </div>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>
                {isNormal
                  ? 'This check did not capture a clear performance problem. If your PC still feels slow, run another check while it is happening.'
                  : processLeader
                    ? `${processLeader.name} was the top ${processResource} user in ${Math.round(processLeader.share_of_named_snapshots * 100)}% of the process checks. Its peak reading was ${processPeak}. This makes it a useful lead to inspect, but does not prove it caused the slowdown.`
                    : causeLimit}
              </p>
            </div>
            <div>
              <div className="metric-label" style={{ marginBottom: '0.35rem' }}>Why CURIO picked this</div>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>{evidenceSummary}</p>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.55, marginTop: '0.5rem' }}>{timingSummary}</p>
              <p style={{ fontSize: '0.76rem', color: 'var(--text-muted)', lineHeight: 1.5, marginTop: '0.45rem' }}>
                CURIO compares this check with examples of normal and problem behavior. Those examples are not a personal history of your computer.
              </p>
            </div>
            <div>
              <div className="metric-label" style={{ marginBottom: '0.35rem' }}>What to check next</div>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', lineHeight: 1.55 }}>{nextStep}</p>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION: WHY CURIO THINKS THIS (Evidence Attribution) */}
      <div className="section-block">
        <h2 className="section-heading">
          <Info size={16} style={{ color: 'var(--accent-zinc)' }} />
          <span>{isNormal ? 'Readings CURIO checked' : 'Clues in the readings'}</span>
        </h2>

        {isNormal ? (
          <div className="discovery-card" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-primary)', fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.4rem' }}>
              <CheckCircle2 size={16} style={{ color: 'var(--status-normal)' }} />
              <span>The readings looked usual during this check.</span>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', lineHeight: 1.6, paddingLeft: '1.5rem' }}>
              CURIO compares your computer’s readings with patterns it has learned from normal and problem conditions. No strong signs of a problem appeared in this check.
            </p>
          </div>
        ) : (
          <div className="evidence-cards-grid">
            {ev.supporting_evidence.map((item, idx) => (
              <div key={idx} className="evidence-card">
                <div className="evidence-feature-header">
                  <div>
                    <div className="evidence-feature-name">{signalName(item.feature_name)}</div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    </div>
                  </div>
                  <span className={`evidence-strength-tag ${item.evidence_strength}`}>
                    {item.evidence_strength === 'strong' ? 'Strong sign' : item.evidence_strength === 'moderate' ? 'Some support' : item.evidence_strength === 'weak' ? 'Small change' : item.evidence_strength}
                  </span>
                </div>

                <div className="evidence-values-row">
                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', display: 'block' }}>This check</span>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {formatSignalValue(item.feature_name, item.observed_value)}
                    </span>
                  </div>

                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', display: 'block' }}>Normal-example level</span>
                    <span style={{ color: 'var(--text-secondary)' }}>
                      {formatSignalValue(item.feature_name, item.normal_reference)}
                    </span>
                  </div>

                  <div>
                    <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem', display: 'block' }}>Change</span>
                    <span style={{ color: item.direction === 'elevated' || item.direction === 'increased' ? 'var(--status-abnormal)' : '#38bdf8' }}>
                      {item.direction === 'elevated' || item.direction === 'increased' ? '↑ Higher' : item.direction === 'reduced' || item.direction === 'decreased' ? '↓ Lower' : '— About the same'}
                    </span>
                  </div>
                </div>

                <p className="evidence-statement-text">
                  {`${signalName(item.feature_name)} was ${item.direction === 'elevated' || item.direction === 'increased' ? 'higher than' : item.direction === 'reduced' || item.direction === 'decreased' ? 'lower than' : 'close to'} its usual level during this check.`}
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
          <span>Changes during the check</span>
        </h2>

        <div className="discovery-card">
          {discStatus === 'OPERATING_EQUILIBRIUM' && (
            <div className="discovery-fallback-banner">
              <strong style={{ color: 'var(--status-normal)', display: 'block', marginBottom: '0.25rem' }}>
                No change in condition
              </strong>
              CURIO did not see a problem develop during this check. Your computer’s readings stayed within its usual range.
            </div>
          )}

          {discStatus === 'SUSTAINED_PRESSURE' && (
            <div className="discovery-fallback-banner" style={{ borderLeft: '3px solid var(--status-abnormal)' }}>
              <strong style={{ color: 'var(--status-abnormal)', display: 'block', marginBottom: '0.25rem' }}>
                The issue was already happening
              </strong>
              The signs of {conditionDisplay.toLowerCase()} were present from the start, so this check could not show when they began.
            </div>
          )}

          {discStatus === 'INSUFFICIENT_TEMPORAL_EVIDENCE' && (
            <div className="discovery-fallback-banner">
              <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.25rem' }}>
                Not enough changes to show a pattern
              </strong>
              CURIO needs to see a few clear changes over time to tell what happened first. This check did not include enough changes.
            </div>
          )}

          {/* Onset Timeline if observed sequence exists */}
          {disc.observed_sequence && disc.observed_sequence.length > 0 && (
            <div style={{ marginTop: '1.25rem' }}>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginBottom: '0.65rem' }}>
                When changes appeared (during this check)
              </div>

              <div className="timeline-track-list">
                {disc.observed_sequence.map((featName, idx) => {
                  const evInfo = disc.onset_events?.find((e) => e.feature_name === featName);
                  const onsetTime = evInfo ? evInfo.onset_time_seconds : 0.0;
                  const positionPct = Math.min(95, Math.max(5, (onsetTime / 30.0) * 100));

                  return (
                    <div key={idx} className="timeline-row">
                      <div className="timeline-feature-label" title={featName}>
                        {signalName(featName)}
                      </div>

                      <div className="timeline-bar-container">
                        <div
                          className="timeline-onset-marker"
                          style={{ left: `${positionPct}%` }}
                        title={`Change first appeared around ${onsetTime.toFixed(1)} seconds`}
                        ></div>
                      </div>

                      <div className="timeline-onset-time">
                        {onsetTime.toFixed(1)}s
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
              <span className="metric-label" title="How closely this check's order of changes matches a known pattern.">Pattern match</span>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: hasValidTau ? 'var(--text-primary)' : 'var(--text-muted)' }}>
                {hasValidTau ? `${Math.round((disc.normalized_tau || 0) * 100)}%` : 'Not enough data'}
              </div>
            </div>

            <div>
              <span className="metric-label" title="How many of the expected readings were available for this analysis.">Readings available</span>
              <div style={{ fontSize: '1.1rem', fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>
                {Math.round(disc.discovery_coverage * 100)}%
              </div>
            </div>

            <div style={{ flex: 1, minWidth: '220px' }}>
              <span className="metric-label">In plain language</span>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontStyle: 'italic', marginTop: '0.15rem' }}>
                {discStatus === 'OPERATING_EQUILIBRIUM' ? 'No problem signs appeared during this check.' : discStatus === 'SUSTAINED_PRESSURE' ? 'The issue was already present when the check began.' : 'There was not enough information to identify a pattern.'}
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
