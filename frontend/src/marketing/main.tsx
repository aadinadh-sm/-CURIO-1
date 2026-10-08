import React, { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, ArrowDown, ArrowRight, ArrowUpRight, Check, CircleDot, Cpu, Database, Download, Eye, HardDrive, Layers3, LockKeyhole, MemoryStick, MoveUpRight, Play, ScanLine, Shield } from 'lucide-react';
import './site.css';

const DOWNLOAD_PATH = 'https://github.com/aadinadh-sm/-CURIO-1/releases/latest/download/curio-windows-preview.zip';

function Mark({ light = false }: { light?: boolean }) {
  return <span className={`cs-mark ${light ? 'cs-mark-light' : ''}`} aria-hidden="true"><Activity size={19} strokeWidth={2.2} /></span>;
}

function Reveal({ children, className = '', delay = 0 }: { children: React.ReactNode; className?: string; delay?: number }) {
  return <div className={`cs-reveal ${className}`} style={{ transitionDelay: `${delay}ms` }}>{children}</div>;
}

function ProductSite() {
  useEffect(() => {
    const nodes = document.querySelectorAll<HTMLElement>('.cs-reveal');
    const observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('cs-visible');
        observer.unobserve(entry.target);
      }
    }), { threshold: 0.12 });
    nodes.forEach((node) => observer.observe(node));
    return () => observer.disconnect();
  }, []);

  const handleTilt = (event: React.PointerEvent<HTMLDivElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width - 0.5;
    const y = (event.clientY - box.top) / box.height - 0.5;
    event.currentTarget.style.setProperty('--tilt-x', `${-y * 4}deg`);
    event.currentTarget.style.setProperty('--tilt-y', `${x * 5}deg`);
  };
  const resetTilt = (event: React.PointerEvent<HTMLDivElement>) => {
    event.currentTarget.style.setProperty('--tilt-x', '0deg');
    event.currentTarget.style.setProperty('--tilt-y', '0deg');
  };

  return (
    <main className="curio-site">
      <nav className="cs-nav">
        <a className="cs-brand" href="#top" aria-label="CURIO home"><Mark light /><span className="cs-brand-name">CURIO<span className="cs-brand-dot">.</span></span></a>
        <div className="cs-nav-links">
          <a href="#how">How it works</a><a href="#inside">Inside CURIO</a><a href="#privacy">Privacy</a>
        </div>
        <div className="cs-nav-right"><a className="cs-nav-download" href="#download">Get CURIO <ArrowUpRight size={14} /></a></div>
      </nav>

      <section className="cs-hero" id="top">
        <div className="cs-hero-grain" />
        <div className="cs-hero-copy">
          <div className="cs-kicker"><span className="cs-kicker-dot" /> ON-DEVICE COMPUTER HEALTH</div>
          <h1>A slow computer<br />is a <span className="cs-title-accent">signal.</span></h1>
          <p className="cs-hero-deck">CURIO looks beyond one busy moment. It follows your computer’s readings over time and shows you what may be worth checking.</p>
          <div className="cs-hero-actions">
            <a className="cs-button cs-button-lime" href="#download"><Download size={17} /> Get CURIO for Windows <ArrowUpRight size={15} /></a>
            <a className="cs-text-button" href="#how"><span className="cs-play-icon"><Play size={11} fill="currentColor" /></span> See how it works</a>
          </div>
          <div className="cs-hero-footnote"><Shield size={14} /> Your readings stay on your computer <span className="cs-footnote-divider" /> No account. No cloud upload.</div>
        </div>

        <div className="cs-hero-visual" onPointerMove={handleTilt} onPointerLeave={resetTilt}>
          <div className="cs-visual-halo" />
          <div className="cs-visual-coordinate cs-mono">FIG. 01 &nbsp; / &nbsp; LIVE SYSTEM TRACE</div>
          <div className="cs-orbit-stage">
            <div className="cs-orbit-shadow" />
            <div className="cs-orbit cs-orbit-a"><i /></div>
            <div className="cs-orbit cs-orbit-b"><i /></div>
            <div className="cs-orbit cs-orbit-c"><i /></div>
            <div className="cs-orbit-core">
              <div className="cs-core-grid" />
              <div className="cs-core-shell" />
              <div className="cs-core-lens"><span /><i /><b /></div>
              <div className="cs-core-glint" />
              <div className="cs-core-label cs-mono"><span>CURIO</span><small>OBSERVING</small></div>
            </div>
            <div className="cs-signal-line cs-signal-line-one"><span /></div>
            <div className="cs-signal-line cs-signal-line-two"><span /></div>
            <div className="cs-signal-line cs-signal-line-three"><span /></div>
            <div className="cs-float-card cs-float-cpu"><div className="cs-float-head"><Cpu size={13} /> PROCESSOR <span className="cs-live-dot" /></div><strong>18<span>%</span></strong><div className="cs-mini-chart"><svg viewBox="0 0 112 28" preserveAspectRatio="none"><path d="M0 21 C10 20 10 13 18 16 S30 24 38 15 S50 8 57 15 S72 22 79 13 S92 8 99 12 S106 10 112 4" /></svg></div></div>
            <div className="cs-float-card cs-float-memory"><div className="cs-float-head"><MemoryStick size={13} /> MEMORY <span className="cs-live-dot" /></div><strong>72<span>%</span></strong><div className="cs-memory-meter"><i /></div><small>11.4 GB in use</small></div>
            <div className="cs-float-card cs-float-disk"><div className="cs-float-head"><HardDrive size={13} /> DISK ACTIVITY</div><strong>3.8 <span>MB/s</span></strong><div className="cs-disk-bars"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div></div>
            <div className="cs-float-tag cs-mono"><span className="cs-tag-pulse" /> PATTERN FOUND <ArrowUpRight size={12} /></div>
          </div>
          <div className="cs-visual-caption cs-mono"><span>01—04</span> CPU / MEMORY / DISK / PROCESS</div>
          <div className="cs-visual-crosshair cs-mono">N 40° 42' 51.2"<br />E 74° 00' 21.8"</div>
        </div>
        <a className="cs-scroll-cue cs-mono" href="#how"><span>SCROLL TO EXPLORE</span><ArrowDown size={14} /></a>
      </section>

      <section className="cs-signal-strip" aria-label="Signals CURIO checks">
        <div className="cs-strip-label cs-mono">THE SIGNALS</div>
        <div className="cs-strip-flow"><span>PROCESSOR LOAD</span><i /> <span>MEMORY PRESSURE</span><i /> <span>DISK ACTIVITY</span><i /> <span>RUNNING APPS</span><i /> <span>CHANGE OVER TIME</span><i /></div>
      </section>

      <section className="cs-section cs-problem" id="how">
        <div className="cs-section-index cs-mono"><span>01</span> / WHY CURIO</div>
        <div className="cs-problem-grid">
          <Reveal><h2>A number tells you<br />what’s busy.<br /><span>Not what to do next.</span></h2></Reveal>
          <Reveal delay={100}><div className="cs-problem-aside"><p>When a PC slows down, Task Manager shows a snapshot. CURIO takes a closer look: it watches CPU, memory, disk and process activity, then compares the readings with patterns it has learned.</p><a href="#inside" className="cs-inline-link">Meet the diagnostic flow <ArrowRight size={15} /></a></div></Reveal>
        </div>
        <div className="cs-three-points">
          <Reveal><article className="cs-point"><span className="cs-point-index cs-mono">01 / CAPTURE</span><div className="cs-point-icon"><ScanLine size={18} /></div><h3>Catch the moment</h3><p>A 30-second check samples your system twice each second and looks for patterns across the readings.</p><div className="cs-point-rule"><i style={{ width: '71%' }} /></div></article></Reveal>
          <Reveal delay={90}><article className="cs-point"><span className="cs-point-index cs-mono">02 / COMPARE</span><div className="cs-point-icon"><Layers3 size={18} /></div><h3>Look for a pattern</h3><p>A calibrated Random Forest compares 12 signals with learned examples of normal, CPU, memory and disk pressure.</p><div className="cs-point-rule"><i style={{ width: '54%' }} /></div></article></Reveal>
          <Reveal delay={180}><article className="cs-point"><span className="cs-point-index cs-mono">03 / EXPLAIN</span><div className="cs-point-icon"><Eye size={18} /></div><h3>Show the evidence</h3><p>See which readings support the result, which point another way, and which app may be worth checking.</p><div className="cs-point-rule"><i style={{ width: '86%' }} /></div></article></Reveal>
        </div>
      </section>

      <section className="cs-instrument" id="inside">
        <div className="cs-instrument-top">
          <div className="cs-section-index cs-mono"><span>02</span> / INSIDE THE CHECK</div>
          <div className="cs-instrument-topnote cs-mono"><CircleDot size={12} /> MODEL · EVIDENCE · TIMELINE</div>
        </div>
        <Reveal className="cs-instrument-heading"><h2>From raw readings<br />to a <span>reasoned lead.</span></h2><p>Three views of the same capture. Each one adds context; none pretends a pattern proves a cause.</p></Reveal>
        <div className="cs-instrument-body">
          <Reveal className="cs-report-wrap">
            <div className="cs-report-card">
              <div className="cs-report-bar"><span className="cs-report-brand"><Mark light /> CURIO <small>LOCAL REPORT</small></span><span className="cs-report-session cs-mono">SESSION 0248 <i /></span></div>
              <div className="cs-report-content">
                <div className="cs-report-eyebrow cs-mono"><span /> POSSIBLE ISSUE FOUND <span className="cs-report-deep">DEEP CHECK · 02:00</span></div>
                <div className="cs-report-main"><div><h3>Memory pressure pattern</h3><p>Memory use stayed elevated through most of this check.</p></div><div className="cs-report-confidence"><small className="cs-mono">MODEL CONFIDENCE</small><strong>84<span>%</span></strong><div><i /></div></div></div>
                <div className="cs-report-mini-grid"><div><small className="cs-mono">PERIODS FLAGGED</small><strong>43 <span>/ 47</span></strong></div><div><small className="cs-mono">SUPPORTING CLUES</small><strong>3 <span>readings</span></strong></div><div><small className="cs-mono">CONFLICTING CLUES</small><strong>1 <span>reading</span></strong></div></div>
                <div className="cs-report-chart-label cs-mono">MEMORY USE THROUGH THE CAPTURE <span>02:00</span></div>
                <div className="cs-report-chart"><div className="cs-chart-y"><span>100</span><span>50</span><span>0</span></div><svg viewBox="0 0 600 90" preserveAspectRatio="none"><path className="cs-chart-area" d="M0 62 C25 58 25 54 48 56 S76 65 92 53 S119 42 138 47 S166 45 184 49 S212 55 230 42 S255 38 276 42 S301 48 320 35 S344 31 367 36 S390 47 412 39 S437 29 457 34 S484 45 504 34 S532 28 548 32 S574 41 600 26 V90 H0Z"/><path className="cs-chart-line" d="M0 62 C25 58 25 54 48 56 S76 65 92 53 S119 42 138 47 S166 45 184 49 S212 55 230 42 S255 38 276 42 S301 48 320 35 S344 31 367 36 S390 47 412 39 S437 29 457 34 S484 45 504 34 S532 28 548 32 S574 41 600 26"/><line x1="0" y1="61" x2="600" y2="61" className="cs-chart-baseline" /></svg><div className="cs-chart-x cs-mono"><span>00:00</span><span>00:30</span><span>01:00</span><span>01:30</span><span>02:00</span></div></div>
                <div className="cs-report-foot"><span><Check size={13} /> READINGS STAY ON DEVICE</span><span className="cs-mono">ILLUSTRATIVE REPORT</span></div>
              </div>
            </div>
            <div className="cs-report-shadow" />
          </Reveal>
          <div className="cs-diagnostic-steps">
            <Reveal><article className="cs-step"><span className="cs-step-number cs-mono">01</span><div><h3>Classify the pattern</h3><p>The model votes across rolling windows, not just one instant. Its confidence is an estimate—not a promise.</p><div className="cs-step-meta cs-mono"><span>RANDOM FOREST</span><span>12 SIGNALS</span></div></div></article></Reveal>
            <Reveal delay={90}><article className="cs-step"><span className="cs-step-number cs-mono">02</span><div><h3>Check the readings</h3><p>Evidence compares measured values with a learned normal reference. CURIO also shows readings that disagree.</p><div className="cs-step-meta cs-mono"><span>Z-SCORE EVIDENCE</span><span>VISIBLE CLUES</span></div></div></article></Reveal>
            <Reveal delay={180}><article className="cs-step"><span className="cs-step-number cs-mono">03</span><div><h3>Follow what changed</h3><p>When several signals shift at different times, CURIO can show their order. If they were already high, it says so.</p><div className="cs-step-meta cs-mono"><span>TIME-SEQUENCE CHECK</span><span>NO CAUSE CLAIMS</span></div></div></article></Reveal>
          </div>
        </div>
      </section>

      <section className="cs-local" id="privacy">
        <div className="cs-local-art" aria-hidden="true"><div className="cs-local-radar"><i /><i /><i /><span /></div><div className="cs-local-pin"><LockKeyhole size={17} /><span>YOUR PC</span></div><div className="cs-local-orbit cs-mono">127.0.0.1 <span>LOCAL</span></div><div className="cs-local-data cs-mono">READINGS<br />● STORED HERE</div></div>
        <Reveal className="cs-local-copy"><div className="cs-section-index cs-mono"><span>03</span> / LOCAL BY DESIGN</div><h2>Your system data<br />belongs <span>to your system.</span></h2><p>CURIO runs on your Windows computer. Live readings and process names stay in its local report history—there’s no account to create and no cloud dashboard.</p><div className="cs-local-facts"><div><LockKeyhole size={16} /><span><strong>On-device</strong><small>Checks run locally</small></span></div><div><Database size={16} /><span><strong>Local history</strong><small>Saved on this computer</small></span></div><div><Shield size={16} /><span><strong>Read-only</strong><small>No system settings changed</small></span></div></div><a className="cs-inline-link" href="#download">See setup details <ArrowRight size={15} /></a></Reveal>
      </section>

      <section className="cs-compare">
        <Reveal><div className="cs-section-index cs-mono"><span>04</span> / A CLEARER NEXT STEP</div><h2>Not a control panel.<br /><span>A place to start.</span></h2></Reveal>
        <div className="cs-compare-grid">
          <Reveal><div className="cs-compare-card cs-compare-muted"><div className="cs-compare-head cs-mono"><span>THE USUAL VIEW</span><span>ONE MOMENT</span></div><div className="cs-compare-number"><b>68</b><small>% CPU</small></div><div className="cs-compare-bars"><i style={{ height: '24%' }} /><i style={{ height: '52%' }} /><i style={{ height: '34%' }} /><i style={{ height: '70%' }} /><i style={{ height: '42%' }} /><i style={{ height: '82%' }} /><i style={{ height: '60%' }} /><i style={{ height: '93%' }} /><i style={{ height: '45%' }} /><i style={{ height: '76%' }} /></div><p>“Something is busy.”</p></div></Reveal>
          <Reveal delay={110}><div className="cs-compare-arrow"><MoveUpRight size={20} /></div></Reveal>
          <Reveal delay={180}><div className="cs-compare-card cs-compare-curio"><div className="cs-compare-head cs-mono"><span>THE CURIO VIEW</span><span>READINGS + CONTEXT</span></div><div className="cs-compare-trace"><span>CPU <i /></span><span>MEMORY <i /></span><span>DISK <i /></span><div className="cs-trace-grid"><i /><i /><i /></div></div><p>“Here’s a pattern—and what to check.”</p></div></Reveal>
        </div>
      </section>

      <section className="cs-download" id="download">
        <div className="cs-download-orbit cs-download-orbit-a" /><div className="cs-download-orbit cs-download-orbit-b" />
        <Reveal className="cs-download-copy"><div className="cs-kicker"><span className="cs-kicker-dot" /> CURIO FOR WINDOWS</div><h2>Get a closer look<br />at your computer.</h2><p>Run a 30-second computer check or analyze a CURIO telemetry CSV. Your data stays on your device.</p><a className="cs-button cs-button-lime cs-download-button" href={DOWNLOAD_PATH} download><Download size={17} /> Download CURIO <ArrowUpRight size={15} /></a><div className="cs-download-meta cs-mono"><span>WINDOWS 10 / 11</span><i /><span>FREE PREVIEW BUILD</span><i /><span>LOCAL-FIRST</span></div><p className="cs-download-requirement">Preview setup bundle. Requires Python 3.12 or 3.13 and an internet connection for first-time setup. No admin access needed.</p></Reveal>
        <div className="cs-download-spec"><div className="cs-spec-line"><span className="cs-mono">01 / UNZIP</span><strong>Extract the download</strong><small>Choose a folder you can find again.</small></div><div className="cs-spec-line"><span className="cs-mono">02 / OPEN</span><strong>Double-click START-CURIO</strong><small>Setup installs its Python packages on first use.</small></div><div className="cs-spec-line"><span className="cs-mono">03 / CHECK</span><strong>Your browser opens the diagnostic app</strong><small>Run a live check, review history, or analyze a CSV.</small></div><div className="cs-spec-version cs-mono"><span>BUILD 1.0.1</span><span>PREVIEW · LOCAL EDITION</span></div></div>
      </section>

      <footer className="cs-footer"><a className="cs-brand" href="#top"><Mark light /><span className="cs-brand-name">CURIO<span className="cs-brand-dot">.</span></span></a><span className="cs-footer-copy">A clearer place to start when your computer feels off.</span><div className="cs-footer-links"><a href="#how">How it works</a><a href="#privacy">Privacy</a><a href="#download">Download</a></div><span className="cs-footer-year cs-mono">© CURIO · LOCAL-FIRST DIAGNOSTICS</span></footer>
    </main>
  );
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><ProductSite /></React.StrictMode>);
