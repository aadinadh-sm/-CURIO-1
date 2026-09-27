import React from 'react';
import { X, Cpu, Layers, HardDrive, Activity, ArrowRight } from 'lucide-react';

interface ReplayModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectReplayCondition: (condition: string) => void;
}

export const ReplayModal: React.FC<ReplayModalProps> = ({
  isOpen,
  onClose,
  onSelectReplayCondition,
}) => {
  if (!isOpen) return null;

  const conditions = [
    {
      id: 'normal',
      name: 'Normal Operation',
      description: 'Replay pre-collected physical baseline telemetry within learned operating bounds.',
      icon: Activity,
      color: '#34d399',
    },
    {
      id: 'cpu_pressure',
      name: 'CPU Pressure',
      description: 'Replay multi-core physical stress telemetry exhibiting sustained high utilization.',
      icon: Cpu,
      color: '#38bdf8',
    },
    {
      id: 'memory_pressure',
      name: 'Memory Pressure',
      description: 'Replay high-RSS memory consumption telemetry with reduced available RAM.',
      icon: Layers,
      color: '#fbbf24',
    },
    {
      id: 'disk_io_pressure',
      name: 'Disk I/O Pressure',
      description: 'Replay high sequential throughput & IOPS telemetry exhibiting storage pressure.',
      icon: HardDrive,
      color: '#f87171',
    },
  ];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" style={{ maxWidth: '620px' }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.85rem' }}>
          <div>
            <div
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.35rem',
                fontSize: '0.68rem',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                fontFamily: 'var(--font-mono)',
                color: '#e4e4e7',
                background: '#18181b',
                border: '1px solid var(--border-subtle)',
                padding: '0.2rem 0.5rem',
                borderRadius: 'var(--radius-xs)',
                marginBottom: '0.45rem',
              }}
            >
              <span>REPLAY / DEMO MODE (DEVELOPMENT TELEMETRY)</span>
            </div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
              Instant Telemetry Replay
            </h2>
          </div>
          <button onClick={onClose} style={{ color: 'var(--text-muted)', cursor: 'pointer', padding: '0.25rem' }}>
            <X size={18} />
          </button>
        </div>

        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.25rem', lineHeight: 1.5 }}>
          Select a verified physical telemetry session to evaluate the full end-to-end diagnosis, evidence, and discovery pipeline instantaneously without waiting 30 seconds.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', marginBottom: '1.75rem' }}>
          {conditions.map((c) => {
            const IconComponent = c.icon;
            return (
              <div
                key={c.id}
                onClick={() => {
                  onSelectReplayCondition(c.id);
                  onClose();
                }}
                style={{
                  background: 'var(--bg-surface-inset)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '0.85rem 1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.85rem',
                  cursor: 'pointer',
                  transition: 'all 0.15s ease',
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-focus)';
                  e.currentTarget.style.background = 'var(--bg-surface-elevated)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-subtle)';
                  e.currentTarget.style.background = 'var(--bg-surface-inset)';
                }}
              >
                <div
                  style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: 'var(--radius-xs)',
                    background: 'var(--bg-surface-elevated)',
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: c.color,
                  }}
                >
                  <IconComponent size={16} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>{c.name}</div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>{c.description}</div>
                </div>
                <div style={{ color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.78rem', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span>Run Replay</span>
                  <ArrowRight size={13} />
                </div>
              </div>
            );
          })}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
          <button className="btn-secondary" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};
