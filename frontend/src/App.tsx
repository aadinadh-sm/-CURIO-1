import React, { useState, useEffect } from 'react';
import { startDiagnosis } from './api';
import { DiagnosisScreen } from './components/DiagnosisScreen';
import { HistoryScreen } from './components/HistoryScreen';
import { HomeScreen } from './components/HomeScreen';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { ReplayModal } from './components/ReplayModal';
import { ResultScreen } from './components/ResultScreen';
import type { DiagnosisResult } from './types';

export const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<'home' | 'diagnosing' | 'result' | 'history'>('home');
  const [activeDiagnosisId, setActiveDiagnosisId] = useState<string | null>(null);
  const [activeResult, setActiveResult] = useState<DiagnosisResult | null>(null);
  const [technicalMode, setTechnicalMode] = useState<boolean>(false);
  const [replayModalOpen, setReplayModalOpen] = useState<boolean>(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  // Keyboard shortcuts (Linear / Raycast pattern: ⌘D for diagnose, ⌘H for history, Esc)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // ⌘D or Ctrl+D for Start Live Diagnosis
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'd') {
        e.preventDefault();
        if (currentView !== 'diagnosing') {
          handleStartLive();
        }
      }
      // ⌘H or Ctrl+H for Incident History
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'h') {
        e.preventDefault();
        setErrorBanner(null);
        setCurrentView('history');
      }
      // Escape for closing replay dialog or clearing errors
      if (e.key === 'Escape') {
        if (replayModalOpen) {
          setReplayModalOpen(false);
        }
        if (errorBanner) {
          setErrorBanner(null);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentView, replayModalOpen, errorBanner]);

  const handleStartLive = async () => {
    setErrorBanner(null);
    try {
      const resp = await startDiagnosis('live');
      setActiveDiagnosisId(resp.diagnosis_id);
      setCurrentView('diagnosing');
    } catch (err: any) {
      setErrorBanner(`Failed to start diagnostic capture: ${err.message}`);
    }
  };

  const handleStartReplay = async (condition: string) => {
    setErrorBanner(null);
    try {
      const resp = await startDiagnosis('replay', condition);
      setActiveDiagnosisId(resp.diagnosis_id);
      setCurrentView('diagnosing');
    } catch (err: any) {
      setErrorBanner(`Failed to start replay diagnosis: ${err.message}`);
    }
  };

  const handleDiagnosisComplete = (result: DiagnosisResult) => {
    setActiveResult(result);
    setCurrentView('result');
  };

  const handleDiagnosisCancelOrError = () => {
    setActiveDiagnosisId(null);
    setCurrentView('home');
  };

  const handleSelectHistoryRecord = (result: DiagnosisResult) => {
    setActiveResult(result);
    setCurrentView('result');
  };

  return (
    <div className="curio-app-shell">
      {/* 240px Desktop Engineering Sidebar */}
      <Sidebar
        currentView={currentView}
        onNavigate={(view) => {
          setErrorBanner(null);
          setCurrentView(view);
        }}
        onStartLiveDiagnosis={handleStartLive}
        onOpenReplay={() => setReplayModalOpen(true)}
        technicalMode={technicalMode}
        onToggleTechnical={() => setTechnicalMode(!technicalMode)}
      />

      {/* Main Workspace with Topbar and Content Area */}
      <div className="curio-workspace">
        <TopBar
          currentView={currentView}
          onNavigate={(view) => {
            setErrorBanner(null);
            setCurrentView(view);
          }}
          onStartLiveDiagnosis={handleStartLive}
          onOpenReplay={() => setReplayModalOpen(true)}
          technicalMode={technicalMode}
          onToggleTechnical={() => setTechnicalMode(!technicalMode)}
        />

        <main className="curio-content">
          {errorBanner && (
            <div
              style={{
                background: 'rgba(244, 63, 94, 0.1)',
                border: '1px solid rgba(244, 63, 94, 0.25)',
                borderRadius: 'var(--radius-md)',
                padding: '0.75rem 1rem',
                color: '#fda4af',
                marginBottom: '1.25rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '0.82rem',
                fontFamily: 'var(--font-mono)',
              }}
            >
              <span>{errorBanner}</span>
              <button
                onClick={() => setErrorBanner(null)}
                style={{ color: '#fda4af', fontWeight: 700, padding: '0 0.25rem' }}
              >
                ✕
              </button>
            </div>
          )}

          {currentView === 'home' && (
            <HomeScreen
              onStartLiveDiagnosis={handleStartLive}
              onOpenHistory={() => setCurrentView('history')}
              onOpenReplay={() => setReplayModalOpen(true)}
              onToggleTechnical={() => setTechnicalMode(!technicalMode)}
              technicalMode={technicalMode}
            />
          )}

          {currentView === 'diagnosing' && activeDiagnosisId && (
            <DiagnosisScreen
              diagnosisId={activeDiagnosisId}
              onComplete={handleDiagnosisComplete}
              onCancel={handleDiagnosisCancelOrError}
              onError={(msg) => setErrorBanner(msg)}
            />
          )}

          {currentView === 'result' && activeResult && (
            <ResultScreen
              result={activeResult}
              onNewDiagnosis={() => setCurrentView('home')}
              onViewHistory={() => setCurrentView('history')}
              defaultTechnicalOpen={technicalMode}
            />
          )}

          {currentView === 'history' && (
            <HistoryScreen
              onSelectRecord={handleSelectHistoryRecord}
              onBackToHome={() => setCurrentView('home')}
            />
          )}
        </main>
      </div>

      {/* Instant Replay Scenario Modal */}
      <ReplayModal
        isOpen={replayModalOpen}
        onClose={() => setReplayModalOpen(false)}
        onSelectReplayCondition={handleStartReplay}
      />
    </div>
  );
};

export default App;
