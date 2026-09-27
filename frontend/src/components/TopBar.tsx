import React from 'react';
import { Play, RotateCcw, Terminal } from 'lucide-react';

interface TopBarProps {
  currentView: 'home' | 'diagnosing' | 'result' | 'history';
  onNavigate: (view: 'home' | 'history') => void;
  onStartLiveDiagnosis: () => void;
  onOpenReplay: () => void;
  technicalMode: boolean;
  onToggleTechnical: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  currentView,
  onNavigate,
  onStartLiveDiagnosis,
  onOpenReplay,
  technicalMode,
  onToggleTechnical,
}) => {
  const getSectionTitle = () => {
    switch (currentView) {
      case 'home':
        return 'Overview';
      case 'diagnosing':
        return 'Diagnostic Workspace';
      case 'result':
        return 'Diagnosis Report';
      case 'history':
        return 'Incident History';
      default:
        return 'Overview';
    }
  };

  return (
    <header className="curio-topbar">
      <div className="topbar-breadcrumb">
        <span
          onClick={() => onNavigate('home')}
          style={{ cursor: 'pointer', transition: 'color 0.12s ease' }}
          title="Return to Overview"
        >
          Workspace
        </span>
        <span>/</span>
        <span className="current">{getSectionTitle()}</span>
      </div>

      <div className="topbar-actions">
        <div className="engine-status-pill">
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#10b981' }}></span>
          <span>Engine: Ready (2 Hz)</span>
        </div>

        {currentView !== 'diagnosing' && (
          <button className="btn-primary" onClick={onStartLiveDiagnosis} style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}>
            <Play size={12} fill="currentColor" />
            <span>Diagnose</span>
            <kbd style={{ background: '#e4e4e7', color: '#18181b', border: '1px solid #d4d4d8' }}>⌘D</kbd>
          </button>
        )}

        <button className="btn-secondary" onClick={onOpenReplay} style={{ fontSize: '0.78rem', padding: '0.35rem 0.75rem' }}>
          <RotateCcw size={12} />
          <span>Replay</span>
        </button>

        <button
          className="btn-secondary"
          onClick={onToggleTechnical}
          style={{
            fontSize: '0.78rem',
            padding: '0.35rem 0.75rem',
            background: technicalMode ? 'rgba(56, 189, 248, 0.1)' : 'var(--bg-surface)',
            color: technicalMode ? '#38bdf8' : 'var(--text-secondary)',
            borderColor: technicalMode ? 'rgba(56, 189, 248, 0.3)' : 'var(--border-subtle)',
          }}
        >
          <Terminal size={12} />
          <span>Technical: {technicalMode ? 'ON' : 'OFF'}</span>
        </button>
      </div>
    </header>
  );
};
