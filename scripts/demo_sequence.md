# CURIO — 2-Minute Capstone Presentation Script

**Target Duration:** Exactly 2 minutes (120 seconds)  
**Presenter:** Student / Candidate  
**Audience:** Examination Committee & Project Evaluators

---

### Step 1: Launch & Local-Only Architecture (0:00 – 0:15)
- **Action**: Open web browser at `http://127.0.0.1:8000` (or `http://127.0.0.1:3000`).
- **Verbal Statement**:
  > *"Good morning committee. Today I present CURIO—an intelligent, local-only computer diagnosis and discovery system. As you can see by this green 'Local Only' badge, CURIO operates strictly on 127.0.0.1. Zero telemetry, user data, or system metrics ever leave this machine."*

### Step 2: Live Telemetry Capture (0:15 – 0:45)
- **Action**: Click the primary `[ Diagnose My Computer ]` button.
- **Verbal Statement**:
  > *"CURIO is now actively sampling our system hardware at 2 Hz for 30 seconds—gathering exactly 61 multi-variate telemetry snapshots across CPU, memory, disk, and process subsystems. This progress bar is driven directly by backend hardware polling events, not an artificial timer."*

### Step 3: Diagnostic Results & Abnormality Boundary (0:45 – 1:00)
- **Action**: Screen transitions to Result View.
- **Verbal Statement**:
  > *"Capture complete! CURIO evaluated 11 rolling feature windows using our Platt-calibrated Random Forest model. It then compared the non-normal probability against an empirical out-of-fold 95th-percentile threshold—determining whether the machine is operating in normal equilibrium or experiencing genuine resource pressure."*

### Step 4: Evidence Engine (Why CURIO Thinks This) (1:00 – 1:15)
- **Action**: Scroll down to the Evidence Cards section.
- **Verbal Statement**:
  > *"Notice that CURIO doesn't just output a class label; it explains WHY. The Evidence Engine computes standardized z-scores and directional alignment relative to our learned Normal baseline. Here we see our top supporting evidence with observed values, baseline baselines, and evidence strength—completely decoupled from raw model confidence."*

### Step 5: Discovery Engine (Temporal Trajectory) (1:15 – 1:30)
- **Action**: Highlight the Discovery Timeline.
- **Verbal Statement**:
  > *"Next is CURIO's distinctive Discovery Engine. It identifies the exact onset order of subsystem disruptions and calculates Kendall's tau-b rank correlation against canonical failure sequences. If the system was already under steady stress, it honestly reports 'Sustained Pressure' without faking an artificial correlation."*

### Step 6: Technical Diagnostics & Replay Demo (1:30 – 1:50)
- **Action**: Click `[ Technical Mode ]` in the header, review the matrix, close it, and click `[ Replay Mode ]` to demo CPU Pressure.
- **Verbal Statement**:
  > *"For examiners, Technical Mode exposes the complete 12-feature matrix, calibrated probability distributions, and Kendall tau statistics. To demonstrate anomalies without waiting 30 seconds, Replay Mode executes the full pipeline over verified physical telemetry in under 400 milliseconds."*

### Step 7: Conclusion & Capstone Summary (1:50 – 2:00)
- **Verbal Statement**:
  > *"Across 22 independent physical sessions, CURIO achieved 95.87% accuracy and a 0.9518 Macro F1, outperforming deterministic rules by +44.9%. Thank you, and I look forward to your questions."*
