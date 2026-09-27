import React, { useEffect, useState } from 'react';
import { ArrowLeft, Search, ChevronRight, FileText } from 'lucide-react';
import { fetchHistory, fetchHistoryDetail } from '../api';
import type { DiagnosisResult, HistorySummaryItem } from '../types';

interface HistoryScreenProps {
  onSelectRecord: (result: DiagnosisResult) => void;
  onBackToHome: () => void;
}

export const HistoryScreen: React.FC<HistoryScreenProps> = ({ onSelectRecord, onBackToHome }) => {
  const [historyList, setHistoryList] = useState<HistorySummaryItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    fetchHistory()
      .then((items) => {
        if (isMounted) {
          setHistoryList(items);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(err.message || 'Failed to load diagnosis history.');
          setLoading(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const handleRowClick = async (sessionId: string) => {
    try {
      const detail = await fetchHistoryDetail(sessionId);
      onSelectRecord(detail);
    } catch (err: any) {
      alert(`Could not load record detail: ${err.message}`);
    }
  };

  const filteredItems = historyList.filter(
    (item) =>
      item.condition.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.session_id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
        <div>
          <h1 style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)', letterSpacing: '-0.02em', marginBottom: '0.25rem' }}>
            Diagnosis History
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
            Recorded diagnostic sessions stored locally on this machine.
          </p>
        </div>

        <button className="btn-secondary" onClick={onBackToHome}>
          <ArrowLeft size={14} />
          <span>Back to Diagnose</span>
        </button>
      </div>

      {/* Search Filter Toolbar */}
      {!loading && historyList.length > 0 && (
        <div style={{ marginBottom: '1rem', display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <div style={{ position: 'relative', width: '280px' }}>
            <Search size={14} style={{ position: 'absolute', left: '0.75rem', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Filter by condition or ID..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                width: '100%',
                background: '#121215',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                padding: '0.45rem 0.75rem 0.45rem 2.2rem',
                color: 'var(--text-primary)',
                fontSize: '0.82rem',
                fontFamily: 'inherit',
                outline: 'none',
              }}
            />
          </div>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            Showing {filteredItems.length} of {historyList.length} records
          </span>
        </div>
      )}

      {loading && (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.88rem' }}>
          Loading historical sessions from local storage...
        </div>
      )}

      {error && (
        <div style={{ padding: '1.5rem', background: 'rgba(244, 63, 94, 0.08)', border: '1px solid rgba(244, 63, 94, 0.25)', borderRadius: 'var(--radius-md)', color: '#fda4af', fontSize: '0.88rem' }}>
          {error}
        </div>
      )}

      {!loading && !error && historyList.length === 0 && (
        <div style={{ background: 'var(--bg-card)', padding: '3rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)', textAlign: 'center' }}>
          <FileText size={32} style={{ color: 'var(--text-muted)', margin: '0 auto 1rem auto', display: 'block' }} />
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1.25rem', fontSize: '0.9rem' }}>
            No previous diagnosis sessions recorded yet.
          </p>
          <button className="btn-primary" onClick={onBackToHome}>
            Run First Diagnosis
          </button>
        </div>
      )}

      {!loading && historyList.length > 0 && (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
          <table className="curio-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Condition</th>
                <th>Confidence</th>
                <th>Status</th>
                <th>Discovery</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredItems.map((item) => {
                const isNormal = item.condition === 'normal';
                const formattedDate = item.timestamp
                  ? new Date(item.timestamp).toLocaleString()
                  : 'N/A';

                return (
                  <tr key={item.session_id} onClick={() => handleRowClick(item.session_id)}>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}>{formattedDate}</td>
                    <td style={{ fontWeight: 600, color: isNormal ? 'var(--text-primary)' : 'var(--text-primary)' }}>
                      {isNormal ? 'Normal Operation' : item.condition.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>
                      {Math.round(item.confidence * 100)}%
                    </td>
                    <td>
                      <span
                        style={{
                          padding: '0.2rem 0.5rem',
                          borderRadius: 'var(--radius-xs)',
                          fontSize: '0.68rem',
                          fontWeight: 700,
                          fontFamily: 'var(--font-mono)',
                          background: item.session_abnormal ? 'rgba(245, 158, 11, 0.1)' : 'rgba(16, 185, 129, 0.1)',
                          color: item.session_abnormal ? '#fbbf24' : '#34d399',
                          border: item.session_abnormal ? '1px solid rgba(245, 158, 11, 0.25)' : '1px solid rgba(16, 185, 129, 0.25)',
                        }}
                      >
                        {item.session_abnormal ? 'ABNORMAL' : 'NORMAL'}
                      </span>
                    </td>
                    <td style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {item.discovery_status}
                    </td>
                    <td>
                      <span style={{ color: 'var(--text-secondary)', fontWeight: 500, fontSize: '0.78rem', display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                        <span>View</span>
                        <ChevronRight size={13} />
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
