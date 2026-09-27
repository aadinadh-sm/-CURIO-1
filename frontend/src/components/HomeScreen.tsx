import React from 'react';
import { fetchHistory, fetchSampleDataset, fetchSystemStatus } from '../api';
import type { HistorySummaryItem, SystemStatus } from '../types';
import {
  Play,
  RotateCcw,
  Clock,
  Terminal,
  Activity,
  Cpu,
  Search,
  GitBranch,
  Layers,
  HardDrive,
  CheckCircle2,
  ArrowRight,
  ShieldCheck,
  Upload,
  FileSpreadsheet,
} from 'lucide-react';

interface HomeScreenProps {
  onStartLiveDiagnosis: () => void;
  onOpenHistory: () => void;
  onOpenReplay: () => void;
  onToggleTechnical: () => void;
  technicalMode: boolean;
  onDiagnoseDataset?: (csv: string, filename: string) => void;
}

export const HomeScreen: React.FC<HomeScreenProps> = ({
  onStartLiveDiagnosis,
  onOpenHistory,
  onOpenReplay,
  onToggleTechnical,
  technicalMode,
  onDiagnoseDataset,
}) => {
  const [recentHistory, setRecentHistory] = React.useState<HistorySummaryItem[]>([]);
  const [systemStatus, setSystemStatus] = React.useState<SystemStatus | null>(null);
  const [datasetBusy, setDatasetBusy] = React.useState(false);
  const [datasetName, setDatasetName] = React.useState<string | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);
  const handleFile = async (file?: File) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv')) return;
    setDatasetName(file.name);
    setDatasetBusy(true);
    try { await onDiagnoseDataset?.(await file.text(), file.name); } finally { setDatasetBusy(false); }
  };
  const handleSample = async (condition: string) => {
    setDatasetBusy(true);
    setDatasetName(`${condition.replace(/_/g, ' ')} sample`);
    try {
      const csv = await fetchSampleDataset(condition);
      await onDiagnoseDataset?.(csv, `curio-${condition}-sample.csv`);
    } finally { setDatasetBusy(false); }
  };
  React.useEffect(() => {
    void Promise.allSettled([fetchHistory(), fetchSystemStatus()]).then(([historyResult, statusResult]) => {
      if (historyResult.status === 'fulfilled') setRecentHistory(historyResult.value.slice(0, 3));
      if (statusResult.status === 'fulfilled') setSystemStatus(statusResult.value);
    });
  }, []);
  return (
    <div className="overview-dashboard">
      {/* Overview Header */}
      <div className="overview-header">
        <div>
          <div className="overview-eyebrow"><span className="eyebrow-rule" /> PERSONAL COMPUTER / FIELD NOTES 01</div>
          <h1 className="overview-title">A clearer picture<br />of your machine.</h1>
          <p className="overview-subtext">
            A quick read on what’s happening under the hood. Private by design, grounded in live system signals.
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button
            className="btn-secondary"
            onClick={onToggleTechnical}
            style={{
              fontSize: '0.75rem',
              padding: '0.35rem 0.7rem',
              color: technicalMode ? '#38bdf8' : 'var(--text-secondary)',
              borderColor: technicalMode ? 'rgba(56, 189, 248, 0.3)' : 'var(--border-subtle)',
            }}
          >
            <Terminal size={12} />
            <span>{technicalMode ? 'Hide technical detail' : 'Show technical detail'}</span>
          </button>
        </div>
      </div>

      <section className="dataset-import-panel" aria-labelledby="dataset-heading">
        <div className="dataset-import-copy">
          <div className="dataset-icon"><FileSpreadsheet size={19} /></div>
          <div>
            <div className="dataset-eyebrow">ANALYZE A CAPTURE</div>
            <h2 id="dataset-heading">Bring your own telemetry</h2>
            <p>Upload a CURIO raw capture. It runs through the same diagnosis, evidence, and timeline analysis as a live check.</p>
          </div>
        </div>
        <div className="dataset-import-action">
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,text/csv"
            className="visually-hidden"
            onChange={(event) => { void handleFile(event.target.files?.[0]); event.currentTarget.value = ''; }}
          />
          <button className="btn-primary dataset-upload-button" disabled={datasetBusy} onClick={() => fileInputRef.current?.click()}>
            <Upload size={15} />
            <span>{datasetBusy ? 'Analyzing capture…' : 'Choose a CSV'}</span>
          </button>
          <span className="dataset-file-note">{datasetName || 'CSV · 61 rows · 30 seconds'}</span>
        </div>
        <div className="dataset-import-footnote">
          <span>Local analysis · File is not retained</span>
          <span>Or diagnose a sample capture:</span>
        </div>
        <div className="dataset-samples">
          {[
            ['normal', 'Normal'],
            ['cpu_pressure', 'CPU pressure'],
            ['memory_pressure', 'Memory pressure'],
            ['disk_io_pressure', 'Disk I/O'],
          ].map(([condition, label]) => (
            <button key={condition} className="sample-chip" disabled={datasetBusy} onClick={() => void handleSample(condition)}>
              {label}
            </button>
          ))}
        </div>
      </section>

      {/* System Status Banner */}
      <div className="overview-status-banner">
        <div className="overview-status-info">
          <div className="overview-status-heading">
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px rgba(16, 185, 129, 0.6)' }}></span>
            <span>Ready when you are</span>
          </div>
          <p className="overview-status-desc">
            The diagnostic engine is standing by. A live check takes about 30 seconds.
          </p>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', marginTop: '0.2rem' }}>
            <ShieldCheck size={13} style={{ color: '#10b981' }} />
            <span style={{ fontSize: '0.72rem', color: '#58775d', fontFamily: 'var(--font-mono)' }}>
              Your data stays on this computer.
            </span>
          </div>
        </div>

        <div className="overview-status-actions">
          <button className="btn-secondary" onClick={onOpenReplay} title="Replay recorded physical stress profiles">
            <RotateCcw size={13} />
            <span>Replay Demo</span>
          </button>

          <button className="btn-primary" onClick={onStartLiveDiagnosis} style={{ padding: '0.55rem 1.15rem' }}>
            <Play size={14} fill="currentColor" />
            <span>Run a system check</span>
            <kbd style={{ background: '#e4e4e7', color: '#18181b', border: '1px solid #d4d4d8', marginLeft: '0.25rem' }}>⌘D</kbd>
          </button>
        </div>
      </div>

      {/* 2-Column Operational Grid: Recent Diagnostics + Contextual System Snapshot */}
      <div className="overview-grid-2col">
        {/* Left Column: Recent Diagnostics */}
        <div className="overview-panel">
          <div className="overview-panel-header">
            <div className="overview-panel-title">
              <Clock size={14} style={{ color: 'var(--text-muted)' }} />
              <span>Recent Diagnostics</span>
            </div>
            <button
              onClick={onOpenHistory}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.25rem',
                fontSize: '0.72rem',
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-muted)',
              }}
            >
              <span>View History</span>
              <ArrowRight size={12} />
            </button>
          </div>

          <div className="recent-diag-list">
            {recentHistory.length ? recentHistory.map((record) => (
              <button className="recent-diag-item" key={record.session_id} onClick={onOpenHistory}>
                <div className="recent-diag-condition">
                  <span className={`status-badge ${record.session_abnormal ? 'abnormal' : 'normal'}`}>{record.condition.replace(/_pressure/g, '').replace(/_/g, ' ').toUpperCase()}</span>
                  <span>{record.condition.replace(/_/g, ' ')}</span>
                </div>
                <span className="recent-diag-meta">{new Date(record.timestamp).toLocaleString()} · {Math.round(record.confidence * 100)}%</span>
              </button>
            )) : (
              <div className="history-empty-note">Your completed checks will appear here. Start a live check or analyze a CSV capture.</div>
            )}
          </div>
        </div>

        {/* Right Column: Contextual System Snapshot */}
        <div className="overview-panel">
          <div className="overview-panel-header">
            <div className="overview-panel-title">
              <Activity size={14} style={{ color: 'var(--text-muted)' }} />
              <span>At a glance</span>
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              {systemStatus?.ready ? 'ENGINE READY' : 'CONNECTING'}
            </span>
          </div>

          <div className="snapshot-metrics-grid">
            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>Model</span>
                <Cpu size={12} />
              </div>
              <div className="snapshot-metric-val">{systemStatus?.model_loaded ? 'Ready' : '—'}</div>
            </div>

            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>Checks run</span>
                <Layers size={12} />
              </div>
              <div className="snapshot-metric-val">{systemStatus?.history_count ?? '—'}</div>
            </div>

            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>Signals</span>
                <HardDrive size={12} />
              </div>
              <div className="snapshot-metric-val" style={{ fontSize: '1rem' }}>
                {systemStatus?.features_count ?? '—'} features
              </div>
            </div>

            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>Alert gate</span>
                <Terminal size={12} />
              </div>
              <div className="snapshot-metric-val">{systemStatus ? `${Math.round(systemStatus.abnormality_threshold * 100)}%` : '—'}</div>
            </div>
          </div>

          <div className="snapshot-subsystem-status">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <CheckCircle2 size={12} style={{ color: '#10b981' }} />
              <span>{systemStatus?.ready ? 'Diagnostic engine online' : 'Waiting for local engine'}</span>
            </div>
            <span>{systemStatus?.classes?.length ?? 4} operating states</span>
          </div>
        </div>
      </div>

      {/* Bottom Section: Diagnostic Core Architecture */}
      <div className="overview-core-section">
        <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', fontWeight: 700, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '0.5px' }}>
          How the check works
        </div>

        <div className="core-cards-grid">
          <div className="core-card">
            <div className="core-card-icon">
              <Clock size={14} />
            </div>
            <div className="core-card-title">30-Second Micro-Capture</div>
            <div className="core-card-desc">
              Captures 61 high-frequency telemetry samples across CPU, RAM, pagefile, and disk subsystems at 2 Hz.
            </div>
          </div>

          <div className="core-card">
            <div className="core-card-icon">
              <Cpu size={14} />
            </div>
            <div className="core-card-title">Calibrated condition model</div>
            <div className="core-card-desc">
              Evaluates 11 rolling feature windows against a calibrated Random Forest classifier with out-of-fold abnormality gating.
            </div>
          </div>

          <div className="core-card">
            <div className="core-card-icon">
              <Search size={14} />
            </div>
            <div className="core-card-title">Evidence Attribution</div>
            <div className="core-card-desc">
              Explains exactly which telemetry signals moved in abnormal directions relative to the learned Normal baseline.
            </div>
          </div>

          <div className="core-card">
            <div className="core-card-icon">
              <GitBranch size={14} />
            </div>
            <div className="core-card-title">Temporal Discovery</div>
            <div className="core-card-desc">
              Identifies chronological onset sequences and computes non-parametric trajectory alignment using Kendall's tau-b.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
