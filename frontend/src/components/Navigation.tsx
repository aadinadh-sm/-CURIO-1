import React from 'react';
import { Activity, Clock, Terminal, RotateCcw } from 'lucide-react';

interface NavigationProps {
  currentView: 'home' | 'diagnosing' | 'result' | 'history';
  onNavigate: (view: 'home' | 'history') => void;
  onOpenReplay: () => void;
  technicalMode: boolean;
  onToggleTechnical: () => void;
}

export const Navigation: React.FC<NavigationProps> = ({
  currentView,
  onNavigate,
  onOpenReplay,
  technicalMode,
  onToggleTechnical,
}) => {
  return (
    <header className="curio-nav">
      <div className="curio-brand" onClick={() => onNavigate('home')}>
        <div className="curio-logo-mark">C</div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
          <span className="curio-title-text">CURIO</span>
          <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            v1.0.0
          </span>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
        <div className="privacy-badge" title="No data leaves this device. Bound strictly to 127.0.0.1.">
          <span className="privacy-dot"></span>
          <span>Local Only</span>
          <span style={{ color: 'var(--text-muted)' }}>127.0.0.1</span>
        </div>

        <div className="curio-nav-actions">
          <button
            className={`nav-link ${currentView === 'home' ? 'active' : ''}`}
            onClick={() => onNavigate('home')}
          >
            <Activity size={14} />
            <span>Diagnose</span>
            <kbd>⌘D</kbd>
          </button>

          <button
            className={`nav-link ${currentView === 'history' ? 'active' : ''}`}
            onClick={() => onNavigate('history')}
          >
            <Clock size={14} />
            <span>History</span>
            <kbd>⌘H</kbd>
          </button>

          <button
            className="nav-link"
            onClick={onOpenReplay}
            style={{ color: '#e4e4e7' }}
            title="Instant replay with verified physical telemetry"
          >
            <RotateCcw size={14} />
            <span>Replay Mode</span>
          </button>

          <button
            className={`nav-link ${technicalMode ? 'active' : ''}`}
            onClick={onToggleTechnical}
            style={{ border: '1px solid var(--border-subtle)' }}
          >
            <Terminal size={14} />
            <span>{technicalMode ? 'Technical: ON' : 'Technical: OFF'}</span>
          </button>
        </div>
      </div>
    </header>
  );
};
