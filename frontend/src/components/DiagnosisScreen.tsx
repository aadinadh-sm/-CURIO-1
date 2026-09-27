import React, { useEffect, useState } from 'react';
import { Activity, AlertCircle, Cpu, HardDrive, Layers, Square, Terminal, XCircle } from 'lucide-react';
import { cancelDiagnosis, fetchDiagnosisStatus } from '../api';
import type { DiagnosisResult, DiagnosisStatusResponse } from '../types';

interface DiagnosisScreenProps {
  diagnosisId: string;
  onComplete: (result: DiagnosisResult) => void;
  onCancel: () => void;
  onError: (errorMsg: string) => void;
}

export const DiagnosisScreen: React.FC<DiagnosisScreenProps> = ({
  diagnosisId,
  onComplete,
  onCancel,
  onError,
}) => {
  const [statusData, setStatusData] = useState<DiagnosisStatusResponse | null>(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const [cancelledMessage, setCancelledMessage] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    let timerId: any = null;

    const poll = async () => {
      try {
        const data = await fetchDiagnosisStatus(diagnosisId);
        if (!isMounted) return;

        setStatusData(data);

        if (data.state === 'complete' && data.result) {
          onComplete(data.result);
          return;
        }

        if (data.state === 'cancelled') {
          setCancelledMessage('Diagnosis Cancelled. System state restored.');
          return;
        }

        if (data.state === 'error') {
          const err = data.error || 'Diagnostic capture encountered an unexpected error.';
          setErrorMessage(err);
          onError(err);
          return;
        }

        // Continue polling every 400ms
        timerId = setTimeout(poll, 400);
      } catch (err: any) {
        if (!isMounted) return;
        setErrorMessage(err.message || 'Lost connection to local CURIO backend.');
        onError(err.message || 'Backend connection error');
      }
    };

    poll();

    return () => {
      isMounted = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [diagnosisId, onComplete, onError]);

  const handleCancelClick = async () => {
    setIsCancelling(true);
    try {
      await cancelDiagnosis(diagnosisId);
      setCancelledMessage('Diagnosis Cancelled. System state restored.');
    } catch {
      setCancelledMessage('Diagnosis Cancelled. System state restored.');
    } finally {
      setIsCancelling(false);
    }
  };

  if (cancelledMessage) {
    return (
      <div className="diagnosis-running-box">
        <div style={{ display: 'inline-flex', padding: '0.75rem', borderRadius: '50%', background: 'rgba(244, 63, 94, 0.1)', border: '1px solid rgba(244, 63, 94, 0.2)', marginBottom: '1.25rem', color: '#fda4af' }}>
          <XCircle size={28} />
        </div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, marginBottom: '0.5rem', color: '#fff', letterSpacing: '-0.02em' }}>
          Diagnosis Cancelled
        </h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.75rem', fontSize: '0.9rem' }}>
          {cancelledMessage}
        </p>
        <button className="btn-secondary" onClick={onCancel}>
          Return Home
        </button>
      </div>
    );
  }

  if (errorMessage) {
    return (
      <div className="diagnosis-running-box">
        <div style={{ display: 'inline-flex', padding: '0.75rem', borderRadius: '50%', background: 'rgba(244, 63, 94, 0.1)', border: '1px solid rgba(244, 63, 94, 0.2)', marginBottom: '1.25rem', color: '#fda4af' }}>
          <AlertCircle size={28} />
        </div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, marginBottom: '0.5rem', color: '#fda4af', letterSpacing: '-0.02em' }}>
          Diagnostic Error
        </h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.75rem', fontSize: '0.9rem' }}>
          {errorMessage}
        </p>
        <button className="btn-secondary" onClick={onCancel}>
          Back to Home
        </button>
      </div>
    );
  }

  const samples = statusData?.samples_collected || 0;
  const total = statusData?.total_samples || 61;
  const progressPercent = Math.round((statusData?.progress || 0) * 100);
  const elapsed = statusData?.elapsed_seconds || 0.0;
  const isAnalyzing = statusData?.state === 'analyzing';

  return (
    <div className="diagnosis-running-box">
      <div
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.45rem',
          padding: '0.25rem 0.65rem',
          borderRadius: 'var(--radius-sm)',
          background: 'rgba(56, 189, 248, 0.08)',
          border: '1px solid rgba(56, 189, 248, 0.25)',
          color: '#38bdf8',
          fontSize: '0.72rem',
          fontWeight: 700,
          fontFamily: 'var(--font-mono)',
          marginBottom: '1rem',
          letterSpacing: '0.5px',
        }}
      >
        <Activity size={12} className="animate-pulse" />
        <span>DIAGNOSIS IN PROGRESS</span>
      </div>

      <h2 style={{ fontSize: '1.75rem', fontWeight: 800, letterSpacing: '-0.02em', marginBottom: '0.4rem', color: '#fff' }}>
        CURIO is analyzing your computer
      </h2>

      <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginBottom: '1.5rem' }}>
        Non-intrusively capturing high-frequency operating telemetry across core subsystems.
      </p>

      {/* Subsystem checklist */}
      <div className="telemetry-checklist">
        <div className={`checklist-item ${samples >= 5 ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Cpu size={14} />
            <span>CPU</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: samples >= 5 ? '#34d399' : 'var(--text-muted)' }}>
            {samples >= 5 ? '✓ Online' : '● Waiting'}
          </span>
        </div>

        <div className={`checklist-item ${samples >= 15 ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Layers size={14} />
            <span>Memory</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: samples >= 15 ? '#34d399' : 'var(--text-muted)' }}>
            {samples >= 15 ? '✓ Online' : '● Waiting'}
          </span>
        </div>

        <div className={`checklist-item ${samples >= 25 ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <HardDrive size={14} />
            <span>Disk I/O</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: samples >= 25 ? '#34d399' : 'var(--text-muted)' }}>
            {samples >= 25 ? '✓ Online' : '● Waiting'}
          </span>
        </div>

        <div className={`checklist-item ${samples >= 35 ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Terminal size={14} />
            <span>Processes</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: samples >= 35 ? '#34d399' : 'var(--text-muted)' }}>
            {samples >= 35 ? '✓ Online' : '● Waiting'}
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${Math.max(5, progressPercent)}%` }}></div>
      </div>

      <div className="progress-stats-row">
        <span>{isAnalyzing ? 'Stage: Analysis' : `Collecting: ${samples} / ${total} samples`}</span>
        <span>{elapsed.toFixed(1)}s / 30.0s ({progressPercent}%)</span>
      </div>

      <div
        style={{
          background: '#0d0d10',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-sm)',
          padding: '0.85rem 1rem',
          marginBottom: '1.75rem',
          textAlign: 'left',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.8rem',
          color: 'var(--text-secondary)',
        }}
      >
        <div style={{ color: '#38bdf8', fontWeight: 600, marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Terminal size={13} />
          <span>{statusData?.stage || 'Initializing telemetry collector...'}</span>
        </div>
        {isAnalyzing && (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>
            Running 12-feature windowing, Random Forest evaluation, evidence attribution, and trajectory progression discovery.
          </div>
        )}
      </div>

      <div>
        <button
          className="btn-danger"
          onClick={handleCancelClick}
          disabled={isCancelling}
        >
          <Square size={13} fill="currentColor" />
          <span>{isCancelling ? 'Restoring System State...' : 'Cancel Diagnosis'}</span>
          <kbd style={{ background: 'rgba(0, 0, 0, 0.4)', color: '#fda4af', border: '1px solid rgba(244, 63, 94, 0.3)' }}>Esc</kbd>
        </button>
      </div>
    </div>
  );
};
