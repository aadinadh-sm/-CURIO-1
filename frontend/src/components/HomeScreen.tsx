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
          <div className="overview-eyebrow"><span className="eyebrow-rule" /> YOUR COMPUTER, MADE CLEARER</div>
          <h1 className="overview-title">Find out why<br />your PC feels slow.</h1>
          <p className="overview-subtext">
            Check your computer for common performance problems. Your readings stay on this computer.
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
            <span>{technicalMode ? 'Hide technical details' : 'Technical details'}</span>
          </button>
        </div>
      </div>

      <section className="dataset-import-panel" aria-labelledby="dataset-heading">
        <div className="dataset-import-copy">
          <div className="dataset-icon"><FileSpreadsheet size={19} /></div>
          <div>
          <div className="dataset-eyebrow">CHECK A CSV FILE</div>
            <h2 id="dataset-heading">Analyze saved computer data</h2>
            <p>Choose a CURIO CSV file to get a diagnosis, see which readings stood out, and review changes over time.</p>
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
            <span>{datasetBusy ? 'Checking file…' : 'Choose a CSV file'}</span>
          </button>
          <span className="dataset-file-note">{datasetName || 'CURIO CSV file'}</span>
        </div>
        <div className="dataset-import-footnote">
          <span>Your file is checked on this computer and is not saved by CURIO.</span>
          <span>Try an example:</span>
        </div>
        <div className="dataset-samples">
          {[
            ['normal', 'Normal'],
            ['cpu_pressure', 'Busy processor'],
            ['memory_pressure', 'Low memory'],
            ['disk_io_pressure', 'Busy disk'],
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
            <span>Ready to check your computer</span>
          </div>
          <p className="overview-status-desc">
            The check takes about 30 seconds. You can keep using your computer while it runs.
          </p>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', marginTop: '0.2rem' }}>
            <ShieldCheck size={13} style={{ color: '#10b981' }} />
            <span style={{ fontSize: '0.72rem', color: '#58775d', fontFamily: 'var(--font-mono)' }}>
              Live reports may show busy app names. Your check details stay on this computer.
            </span>
          </div>
        </div>

        <div className="overview-status-actions">
          <button className="btn-secondary" onClick={onOpenReplay} title="View an example result">
            <RotateCcw size={13} />
            <span>View an example</span>
          </button>

          <button className="btn-primary" onClick={onStartLiveDiagnosis} style={{ padding: '0.55rem 1.15rem' }}>
            <Play size={14} fill="currentColor" />
            <span>Check my computer</span>
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
              <span>Recent checks</span>
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
              <span>See all</span>
              <ArrowRight size={12} />
            </button>
          </div>

          <div className="recent-diag-list">
            {recentHistory.length ? recentHistory.map((record) => (
              <button className="recent-diag-item" key={record.session_id} onClick={onOpenHistory}>
                <div className="recent-diag-condition">
                  <span className={`status-badge ${record.session_abnormal ? 'abnormal' : 'normal'}`}>{record.session_abnormal ? 'CHECK NEEDED' : 'LOOKS OK'}</span>
                  <span>{{ normal: 'No problem found', cpu_pressure: 'CPU under heavy load', memory_pressure: 'Memory running low', disk_io_pressure: 'Disk under heavy load' }[record.condition] || record.condition.replace(/_/g, ' ')}</span>
                </div>
                <span className="recent-diag-meta">{new Date(record.timestamp).toLocaleString()} · {Math.round(record.confidence * 100)}%</span>
              </button>
            )) : (
              <div className="history-empty-note">Your recent results will appear here after you check your computer or analyze a CSV file.</div>
            )}
          </div>
        </div>

        {/* Right Column: Contextual System Snapshot */}
        <div className="overview-panel">
          <div className="overview-panel-header">
            <div className="overview-panel-title">
              <Activity size={14} style={{ color: 'var(--text-muted)' }} />
              <span>CURIO status</span>
            </div>
            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              {systemStatus?.ready ? 'READY' : 'CONNECTING'}
            </span>
          </div>

          <div className="snapshot-metrics-grid">
            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>Checker</span>
                <Cpu size={12} />
              </div>
              <div className="snapshot-metric-val">{systemStatus?.model_loaded ? 'Ready' : '—'}</div>
            </div>

            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>Checks so far</span>
                <Layers size={12} />
              </div>
              <div className="snapshot-metric-val">{systemStatus?.history_count ?? '—'}</div>
            </div>

            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>Computer parts checked</span>
                <HardDrive size={12} />
              </div>
              <div className="snapshot-metric-val" style={{ fontSize: '1rem' }}>
                CPU, memory, disk
              </div>
            </div>

            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>Problem types</span>
                <Terminal size={12} />
              </div>
              <div className="snapshot-metric-val">{systemStatus?.classes?.length ?? 4}</div>
            </div>
          </div>

          <div className="snapshot-subsystem-status">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <CheckCircle2 size={12} style={{ color: '#10b981' }} />
              <span>{systemStatus?.ready ? 'Ready to check your computer' : 'Starting CURIO…'}</span>
            </div>
            <span>Results stay on this computer</span>
          </div>
        </div>
      </div>

      {/* Bottom Section: Diagnostic Core Architecture */}
      <div className="overview-core-section">
        <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', fontWeight: 700, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', letterSpacing: '0.5px' }}>
          How CURIO checks your computer
        </div>

        <div className="core-cards-grid">
          <div className="core-card">
            <div className="core-card-icon">
              <Clock size={14} />
            </div>
            <div className="core-card-title">1. Checks key parts</div>
            <div className="core-card-desc">
              For about 30 seconds, CURIO checks your processor, memory, storage, and running apps.
            </div>
          </div>

          <div className="core-card">
            <div className="core-card-icon">
              <Cpu size={14} />
            </div>
            <div className="core-card-title">2. Looks for known problems</div>
            <div className="core-card-desc">
              It compares the readings with examples of normal use and common performance problems.
            </div>
          </div>

          <div className="core-card">
            <div className="core-card-icon">
              <Search size={14} />
            </div>
            <div className="core-card-title">3. Shows what stood out</div>
            <div className="core-card-desc">
              The report points out which readings changed and how they compare with typical levels.
            </div>
          </div>

          <div className="core-card">
            <div className="core-card-icon">
              <GitBranch size={14} />
            </div>
            <div className="core-card-title">4. Checks what changed first</div>
            <div className="core-card-desc">
              When there is enough data, CURIO shows when changes appeared and whether they followed a familiar pattern.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
