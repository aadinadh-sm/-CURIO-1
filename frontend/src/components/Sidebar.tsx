import React from 'react';
import { Activity, LayoutDashboard, Play, Clock, Cpu, Layers, HardDrive, Terminal, Code2, RotateCcw } from 'lucide-react';

interface SidebarProps {
  currentView: 'home' | 'diagnosing' | 'result' | 'history';
  onNavigate: (view: 'home' | 'history') => void;
  onStartLiveDiagnosis?: () => void;
  onOpenReplay: () => void;
  technicalMode: boolean;
  onToggleTechnical: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentView,
  onNavigate,
  onStartLiveDiagnosis,
  onOpenReplay,
  technicalMode,
  onToggleTechnical,
}) => {
  return (
    <aside className="curio-sidebar">
      <div>
        {/* Brand Header */}
        <div className="sidebar-header" onClick={() => onNavigate('home')} style={{ cursor: 'pointer' }}>
          <div className="sidebar-brand-mark" aria-hidden="true">
            <Activity size={21} strokeWidth={2.1} />
            <span className="sidebar-brand-signal" />
          </div>
          <div className="sidebar-brand-copy">
            <div className="sidebar-brand-title">CURIO</div>
            <div className="sidebar-brand-subtitle">
              SYSTEM INTELLIGENCE
            </div>
          </div>
        </div>

        {/* Primary Navigation */}
        <div className="sidebar-section-title">Navigation</div>
        <div className="sidebar-nav-group">
          <button
            className={`sidebar-nav-item ${currentView === 'home' ? 'active' : ''}`}
            onClick={() => onNavigate('home')}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <LayoutDashboard size={15} />
              <span>Overview</span>
            </div>
          </button>

          <button
            className={`sidebar-nav-item ${currentView === 'diagnosing' ? 'active' : ''}`}
            onClick={() => {
              if (onStartLiveDiagnosis) onStartLiveDiagnosis();
              else onNavigate('home');
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Play size={15} />
              <span>Diagnose</span>
            </div>
            <kbd>⌘D</kbd>
          </button>

          <button
            className={`sidebar-nav-item ${currentView === 'history' ? 'active' : ''}`}
            onClick={() => onNavigate('history')}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Clock size={15} />
              <span>History</span>
            </div>
            <kbd>⌘H</kbd>
          </button>
        </div>

        {/* Subsystems (Monitoring Context) */}
        <div className="sidebar-section-title">System</div>
        <div className="sidebar-nav-group">
          <div className="sidebar-subsystem-item">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Cpu size={14} />
              <span>CPU</span>
            </div>
            <span style={{ fontSize: '0.7rem', color: '#34d399' }}>● Active</span>
          </div>

          <div className="sidebar-subsystem-item">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Layers size={14} />
              <span>Memory</span>
            </div>
            <span style={{ fontSize: '0.7rem', color: '#34d399' }}>● Active</span>
          </div>

          <div className="sidebar-subsystem-item">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <HardDrive size={14} />
              <span>Disk</span>
            </div>
            <span style={{ fontSize: '0.7rem', color: '#34d399' }}>● Active</span>
          </div>

          <div className="sidebar-subsystem-item">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Terminal size={14} />
              <span>Processes</span>
            </div>
            <span style={{ fontSize: '0.7rem', color: '#34d399' }}>● Active</span>
          </div>
        </div>

        {/* Technical Mode Toggle */}
        <div className="sidebar-section-title">Console</div>
        <div className="sidebar-nav-group">
          <button
            className={`sidebar-nav-item ${technicalMode ? 'active' : ''}`}
            onClick={onToggleTechnical}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Code2 size={15} />
              <span>Technical Mode</span>
            </div>
            <span
              style={{
                fontSize: '0.62rem',
                fontFamily: 'var(--font-mono)',
                padding: '0.1rem 0.35rem',
                borderRadius: '3px',
                background: technicalMode ? 'rgba(56, 189, 248, 0.15)' : 'var(--bg-surface-inset)',
                color: technicalMode ? '#38bdf8' : 'var(--text-secondary)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              {technicalMode ? 'ON' : 'OFF'}
            </span>
          </button>

          <button
            className="sidebar-nav-item"
            onClick={onOpenReplay}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <RotateCcw size={15} />
              <span>Replay Demo</span>
            </div>
            <span style={{ fontSize: '0.62rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
              FIXTURE
            </span>
          </button>
        </div>
      </div>

      {/* Sidebar Footer */}
      <div className="sidebar-footer">
        <div className="sidebar-local-status" title="Bound strictly to 127.0.0.1. Zero outbound telemetry.">
          <div className="status-dot-emerald"></div>
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)', lineHeight: 1.2 }}>Local Only</span>
            <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', lineHeight: 1.2 }}>Telemetry stays here</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
