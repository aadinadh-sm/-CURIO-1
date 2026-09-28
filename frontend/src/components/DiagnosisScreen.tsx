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
          setCancelledMessage('The check was stopped. CURIO did not change your system settings.');
          return;
        }

        if (data.state === 'error') {
          const err = 'CURIO could not finish the check. Try again. If this keeps happening, restart CURIO.';
          setErrorMessage(err);
          onError(err);
          return;
        }

        // Continue polling every 400ms
        timerId = setTimeout(poll, 400);
      } catch (err: any) {
        if (!isMounted) return;
        const friendlyError = 'CURIO lost its connection while checking your computer. Restart CURIO and try again.';
        setErrorMessage(friendlyError);
        onError(friendlyError);
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
      setCancelledMessage('The check was stopped. CURIO did not change your system settings.');
    } catch {
      setCancelledMessage('The check was stopped. CURIO did not change your system settings.');
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
        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, marginBottom: '0.5rem', color: 'var(--text-primary)', letterSpacing: '-0.02em' }}>
          Check stopped
        </h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.75rem', fontSize: '0.9rem' }}>
          {cancelledMessage}
        </p>
        <button className="btn-secondary" onClick={onCancel}>
          Back to CURIO
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
          Check could not finish
        </h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.75rem', fontSize: '0.9rem' }}>
          {errorMessage}
        </p>
        <button className="btn-secondary" onClick={onCancel}>
          Back to CURIO
        </button>
      </div>
    );
  }

  const samples = statusData?.samples_collected || 0;
  const total = statusData?.total_samples || 61;
  const progressPercent = Math.round((statusData?.progress || 0) * 100);
  const elapsed = statusData?.elapsed_seconds || 0.0;
  const isAnalyzing = statusData?.state === 'analyzing';
  const stageLabel = isAnalyzing
    ? 'Reviewing the readings'
    : samples === 0
      ? 'Getting ready to check your computer'
      : `Checking your computer (${samples} of ${total} readings)`;

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
        <span>CHECK IN PROGRESS</span>
      </div>

      <h2 style={{ fontSize: '1.75rem', fontWeight: 800, letterSpacing: '-0.02em', marginBottom: '0.4rem', color: 'var(--text-primary)' }}>
        Checking your computer
      </h2>

      <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginBottom: '1.5rem' }}>
        CURIO is checking your processor, memory, storage, and running apps. This usually takes about 30 seconds.
      </p>

      {/* Subsystem checklist */}
      <div className="telemetry-checklist">
        <div className={`checklist-item ${samples >= 5 ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Cpu size={14} />
            <span>CPU</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: samples >= 5 ? '#34d399' : 'var(--text-muted)' }}>
            {samples >= 5 ? '✓ Checked' : '● Checking soon'}
          </span>
        </div>

        <div className={`checklist-item ${samples >= 15 ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Layers size={14} />
            <span>Memory</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: samples >= 15 ? '#34d399' : 'var(--text-muted)' }}>
            {samples >= 15 ? '✓ Checked' : '● Checking soon'}
          </span>
        </div>

        <div className={`checklist-item ${samples >= 25 ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <HardDrive size={14} />
            <span>Disk I/O</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: samples >= 25 ? '#34d399' : 'var(--text-muted)' }}>
            {samples >= 25 ? '✓ Checked' : '● Checking soon'}
          </span>
        </div>

        <div className={`checklist-item ${samples >= 35 ? 'active' : ''}`}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Terminal size={14} />
            <span>Processes</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: samples >= 35 ? '#34d399' : 'var(--text-muted)' }}>
            {samples >= 35 ? '✓ Checked' : '● Checking soon'}
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${Math.max(5, progressPercent)}%` }}></div>
      </div>

      <div className="progress-stats-row">
        <span>{stageLabel}</span>
        <span>{elapsed.toFixed(1)}s / 30.0s ({progressPercent}%)</span>
      </div>

      <div
        style={{
          background: 'var(--bg-surface-inset)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-sm)',
          padding: '0.85rem 1rem',
          marginBottom: '1.75rem',
          textAlign: 'left',
          fontFamily: 'inherit',
          fontSize: '0.8rem',
          color: 'var(--text-secondary)',
        }}
      >
        <div style={{ color: '#38bdf8', fontWeight: 600, marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Terminal size={13} />
          <span>{stageLabel}</span>
        </div>
        {isAnalyzing && (
          <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>
            CURIO is comparing the readings and preparing your report.
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
          <span>{isCancelling ? 'Stopping check…' : 'Stop check'}</span>
          <kbd style={{ background: 'rgba(0, 0, 0, 0.4)', color: '#fda4af', border: '1px solid rgba(244, 63, 94, 0.3)' }}>Esc</kbd>
        </button>
      </div>
    </div>
  );
};
