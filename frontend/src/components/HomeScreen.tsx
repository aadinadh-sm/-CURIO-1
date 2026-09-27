import React from 'react';
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
} from 'lucide-react';

interface HomeScreenProps {
  onStartLiveDiagnosis: () => void;
  onOpenHistory: () => void;
  onOpenReplay: () => void;
  onToggleTechnical: () => void;
  technicalMode: boolean;
}

export const HomeScreen: React.FC<HomeScreenProps> = ({
  onStartLiveDiagnosis,
  onOpenHistory,
  onOpenReplay,
  onToggleTechnical,
  technicalMode,
}) => {
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
            <span style={{ fontSize: '0.72rem', color: '#a7f3d0', fontFamily: 'var(--font-mono)' }}>
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
            <div className="recent-diag-item" onClick={onOpenHistory}>
              <div className="recent-diag-condition">
                <span className="status-badge normal">NORMAL</span>
                <span>Normal Operating State</span>
              </div>
              <span className="recent-diag-meta">Recent • 30.0s</span>
            </div>

            <div className="recent-diag-item" onClick={onOpenHistory}>
              <div className="recent-diag-condition">
                <span className="status-badge abnormal">CPU</span>
                <span>CPU Pressure Incident</span>
              </div>
              <span className="recent-diag-meta">Verified • 96% conf</span>
            </div>

            <div className="recent-diag-item" onClick={onOpenHistory}>
              <div className="recent-diag-condition">
                <span className="status-badge abnormal">DISK</span>
                <span>Disk I/O Contention</span>
              </div>
              <span className="recent-diag-meta">Prior • Sustained</span>
            </div>
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
              2 Hz Polling Ready
            </span>
          </div>

          <div className="snapshot-metrics-grid">
            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>CPU Load</span>
                <Cpu size={12} />
              </div>
              <div className="snapshot-metric-val">18%</div>
            </div>

            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>RAM Usage</span>
                <Layers size={12} />
              </div>
              <div className="snapshot-metric-val">62%</div>
            </div>

            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>Disk State</span>
                <HardDrive size={12} />
              </div>
              <div className="snapshot-metric-val" style={{ fontSize: '1rem', color: '#34d399' }}>
                idle
              </div>
            </div>

            <div className="snapshot-metric-card">
              <div className="snapshot-metric-label">
                <span>Processes</span>
                <Terminal size={12} />
              </div>
              <div className="snapshot-metric-val">184</div>
            </div>
          </div>

          <div className="snapshot-subsystem-status">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <CheckCircle2 size={12} style={{ color: '#10b981' }} />
              <span>Subsystems Operational</span>
            </div>
            <span>12 Frozen Features</span>
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
            <div className="core-card-title">A baseline built for your machine</div>
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
