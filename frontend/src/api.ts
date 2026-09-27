import type {
  DiagnosisResult,
  DiagnosisStatusResponse,
  HistorySummaryItem,
  SystemStatus,
} from './types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<{ status: string; local_only: boolean }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`);
  return res.json();
}

export async function startDiagnosis(
  mode: 'live' | 'replay' = 'live',
  replayCondition?: string,
  sessionId?: string
): Promise<{ diagnosis_id: string; state: string }> {
  const res = await fetch(`${API_BASE}/diagnosis/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mode,
      replay_condition: replayCondition,
      session_id: sessionId,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to start diagnosis');
  }
  return res.json();
}

export async function fetchDiagnosisStatus(diagnosisId: string): Promise<DiagnosisStatusResponse> {
  const res = await fetch(`${API_BASE}/diagnosis/${encodeURIComponent(diagnosisId)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to fetch diagnosis status');
  }
  return res.json();
}

export async function cancelDiagnosis(diagnosisId: string): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/diagnosis/${encodeURIComponent(diagnosisId)}/cancel`, {
    method: 'POST',
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to cancel diagnosis');
  }
  return res.json();
}

export async function fetchHistory(): Promise<HistorySummaryItem[]> {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error(`Failed to load history: ${res.statusText}`);
  return res.json();
}

export async function fetchHistoryDetail(sessionId: string): Promise<DiagnosisResult> {
  const res = await fetch(`${API_BASE}/history/${encodeURIComponent(sessionId)}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Historical record not found');
  }
  return res.json();
}
