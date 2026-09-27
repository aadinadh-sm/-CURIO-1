# CURIO — Viva & Defense Examination Guide

**Project:** CURIO — Intelligent Machine Learning Discovery and Computer Diagnosis System  
**Prepared for:** Undergraduate Capstone Examination & Viva Defense

---

### 1. What problem does CURIO solve?
Standard operating system monitors (such as Windows Task Manager or Linux `top`) display instantaneous percentages of resource consumption without context, temporal trajectory, or historical baselines. Enterprise APM suites require continuous cloud agent streaming, user tracking, and subscription fees. Everyday users and technicians are left wondering: *"Why is the system sluggish, what changed first, and what concrete evidence supports this diagnosis?"* CURIO provides a local-only, privacy-preserving assistant that takes a short 30-second capture, classifies resource pressure, explains why using statistical evidence attribution, and uncovers the chronological progression of subsystem degradation.

---

### 2. Why is this Machine Learning?
Operating system metrics are deeply multivariate, highly interdependent, and noisy. For example, high disk I/O could indicate normal video rendering, or it could be pagefile thrashing caused by low RAM. Static hand-tuned thresholds fail because acceptable utilization varies dynamically depending on concurrent background workloads and hardware capacity. Machine learning discovers non-linear decision boundaries across multi-dimensional feature spaces, recognizing complex combinations of core imbalance, swap activity, and context switches that heuristic rules consistently misclassify.

---

### 3. Why Random Forest?
Random Forest was chosen because:
1. **Tabular Performance**: Extensive empirical research has demonstrated that tree-based ensembles (such as Random Forest and Gradient Boosted Trees) consistently outperform neural networks on tabular datasets with heterogeneous numerical features.
2. **Robustness to Feature Scales**: Decision trees split on individual feature values, making them invariant to monotonic scaling differences between percentages (0–100%), normalized rates (0–10), and raw counts.
3. **No Overfitting on Small Datasets**: Bagging (Bootstrap Aggregation) and random feature subsampling reduce model variance, preventing the classifier from memorizing specific background process quirks.
4. **Sub-millisecond Inference**: Evaluating 100 shallow decision trees across 11 windows requires under 15 milliseconds on standard commodity CPUs without GPU acceleration.

---

### 4. Why not a Neural Network?
1. **Data Efficiency**: Deep neural networks require tens of thousands of diverse training instances to avoid severe overfitting. With 242 physical feature windows, a deep neural network would readily memorize session artifacts.
2. **No GPU Requirement**: CURIO is designed to run efficiently on low-power, constrained desktop environments. Neural architectures introduce large runtime runtimes (e.g. PyTorch/TensorFlow, 500+ MB dependencies), whereas scikit-learn is lightweight.
3. **Interpretability & Determinism**: Tree-based ensembles are deterministic, inspectable, and produce well-behaved out-of-fold probability estimates.

---

### 5. What are the 12 features?
The 12 features capture the dynamic health of CPU, memory, disk, and process subsystems:
1. `cpu_mean`: Average CPU utilization across all logical cores (0–100%).
2. `cpu_std`: Temporal volatility / fluctuation of CPU utilization.
3. `cpu_max`: Peak instantaneous core utilization observed within the window.
4. `cpu_core_imbalance`: Difference between the most utilized and least utilized core, detecting single-threaded bottlenecks.
5. `top_proc_cpu_ratio`: Ratio of total CPU time consumed by the single heaviest process.
6. `ram_used_pct`: Percentage of physical RAM allocated.
7. `ram_available_ratio`: Ratio of unallocated/cache-evictable memory to total installed capacity.
8. `swap_used_pct`: Percentage of virtual pagefile/swap space in use.
9. `disk_io_rate_norm`: Normalized combined read/write throughput (MB/s log-scaled).
10. `disk_iops_norm`: Normalized combined I/O operations per second (IOPS log-scaled).
11. `process_count_delta`: Net change in active OS process count over the window.
12. `top_proc_mem_pct`: Physical RAM percentage consumed by the dominant memory process.

---

### 6. Why 30 seconds?
A 30-second capture window represents an optimal trade-off between user patience and temporal resolution:
- 5 or 10 seconds is too short to observe genuine system stress cascades (e.g., RAM exhaustion leading to pagefile thrashing).
- 2 to 5 minutes creates excessive user friction during interactive diagnosis.
- In 30 seconds at 2 Hz, exactly 61 samples are gathered—providing sufficient signal to observe onset transitions while maintaining near-instant responsiveness.

---

### 7. Why 11 windows?
Using a 5.0-second rolling window with a 2.5-second step interval over a 30.0-second session yields exactly 11 distinct temporal windows:
$$\text{Windows} = \frac{30.0 - 5.0}{2.5} + 1 = 11$$
This sliding window preserves temporal continuity, smooths high-frequency sampling jitter, and allows the Discovery Engine to track how conditions evolve from Window 0 ($t = 0.0\text{s} - 5.0\text{s}$) to Window 10 ($t = 25.0\text{s} - 30.0\text{s}$).

---

### 8. Why session-level grouping?
In time-series and sensor telemetry, consecutive rolling windows from the same 30-second capture share temporal autocorrelation. If random K-Fold cross-validation were used, windows from the same capture session would appear in both training and test folds, causing severe data leakage and artificially inflated accuracy. GroupKFold grouped by `session_id` ensures that all 11 windows of a session remain strictly together in either the training set or the test set.

---

### 9. Why calibration?
Raw Random Forest probability outputs (the fraction of trees voting for a class) are notoriously uncalibrated: they tend to push probabilities toward 0.0 or 1.0, or bunch around intermediate values. CURIO fits a Platt sigmoid calibration mapping:
$$P(\text{condition} \mid f) = \frac{1}{1 + \exp(-(A \cdot f + B))}$$
This transforms raw voting tallies into statistically meaningful posterior probabilities that reflect true empirical risk.

---

### 10. What is abnormality_score?
The abnormality score measures how strongly a session departs from the Normal operating condition:
$$\text{abnormality\_score} = 1.0 - P(\text{normal})$$
It aggregates the non-normal probability mass across the session. If the mean abnormality score falls below the deployment threshold, the session is classified as Normal, preventing false alarms.

---

### 11. Why the 95th percentile?
The abnormality threshold (`0.8536`) was determined strictly from the out-of-fold cross-validation distribution of verified Normal sessions during training. Setting the threshold at the 95th percentile ensures that by mathematical definition, 95% of normal operating variations will be accepted as normal, capping the false-positive rate at 5%.

---

### 12. What is the Evidence Engine?
The Evidence Engine explains **WHY** CURIO made a diagnosis. For the representative strongest window, it calculates:
1. Standardized distance ($z$-score) relative to the training-only Normal baseline: $z = \frac{x - \mu_{\text{normal}}}{\sigma_{\text{normal}}}$.
2. Directional alignment: checks whether the feature moved in the expected direction (+1 for elevation, -1 for depression) according to condition domain physics.
3. Evidence strength categorization (`Strong`, `Moderate`, `Inconclusive`, `Contradictory`).

---

### 13. What is the Discovery Engine?
The Discovery Engine discovers **HOW** the condition unfolded over time. It identifies the exact onset timestamp ($t_{\text{onset}}$) when each canonical feature deviated by more than $1.5\sigma$ from baseline, establishes the observed temporal sequence, and calculates rank correlation against expected canonical failure cascades.

---

### 14. What is Kendall's tau ($\tau$)?
Kendall's tau-b ($\tau$) is a non-parametric statistic that measures the rank correlation / ordinal agreement between two ordered sequences:
$$\tau = \frac{P - Q}{\sqrt{(P + Q + T) \cdot (P + Q + U)}}$$
Where $P$ is concordant pairs and $Q$ is discordant pairs. $\tau = +1.0$ indicates perfect chronological agreement with the canonical progression, $\tau = 0.0$ indicates random ordering, and $\tau = -1.0$ indicates reversed ordering.

---

### 15. Does Kendall tau prove causality?
**No.** Kendall's tau establishes rank correlation in observed temporal sequence; it does not prove physical or logical causality. In operating systems, two independent subsystems may escalate concurrently due to an unobserved third factor (confounding). CURIO strictly uses descriptive, non-causal language and never claims root-cause determination.

---

### 16. Why does Memory Pressure have lower separation?
In modern operating systems with aggressive caching, "free" RAM is rarely high; the OS uses almost all spare memory for disk buffers and page caches. Furthermore, memory allocation occurs in phases: working sets grow in RAM first, and only when physical RAM is exhausted does pagefile paging occur. Thus, mild memory pressure overlaps significantly with normal multi-tab browser usage, making classification boundaries subtler than CPU or Disk saturation.

---

### 17. Why is cross-machine validation a limitation?
Hardware differences (CPU instruction sets, core counts, RAM capacities, SSD vs HDD transfer speeds, kernel schedulers) alter the baseline operating points and variance of raw telemetry metrics. Because physical evaluation was performed on Machine A, CURIO's models and reference profiles are calibrated to Machine A's hardware environment. Claiming universal cross-hardware generalization without evaluating distinct physical architectures would be scientifically unsound.

---

### 18. What happens if there is insufficient temporal evidence?
If fewer than 2 canonical features exhibit distinct onset transitions during the 30-second window, or if the stress was already present at $t = 0\text{s}$, CURIO falls back cleanly:
- It returns `SUSTAINED_PRESSURE` (if stress was constant throughout).
- It returns `INSUFFICIENT_TEMPORAL_EVIDENCE` (if transitions were absent).
- It suppresses the $\tau$ metric rather than rendering an artificial or misleading score.

---

### 19. What is the difference between Evidence and Discovery?
- **Evidence Engine** is **static / cross-sectional**: it examines the *magnitude* of feature deviations at the representative window to explain *why* the condition was recognized.
- **Discovery Engine** is **dynamic / longitudinal**: it examines the *timing and order* of deviations across all 11 windows to explain *how* the condition progressed.

---

### 20. What happens if the model artifact is missing?
The backend fails gracefully:
- `/api/status` returns `ready: false` with a clear human-readable error.
- Any attempt to start a diagnosis returns HTTP 503 Service Unavailable detailing the missing artifact.
- The UI displays an informative alert prompting the administrator to run `python scripts/build_deployment_model.py`.
- No raw Python stack traces are exposed to users.

---

### 21. Why is CURIO local-only?
Operating system telemetry captures sensitive private information: running process names, disk I/O activity, memory allocation patterns, and system timestamps. Transmitting this data over the network creates privacy vulnerabilities, compliance risks (GDPR, HIPAA), and potential attack vectors. By binding strictly to `127.0.0.1` and executing all ML inference locally, CURIO guarantees zero data egress and zero cloud dependence.
