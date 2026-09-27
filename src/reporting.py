"""Reporting Module for CURIO Live Diagnosis Pipeline.

Renders human-readable diagnosis reports in standard executive mode and deep
technical diagnostic mode, adhering strictly to non-causal language and
ASCII-safe formatting across standard terminal consoles.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.evidence import validate_no_causal_language


def _format_confidence_pct(val: float) -> str:
    """Formats a float probability [0.0, 1.0] as percentage string."""
    return f"{val * 100.0:.1f}%"


def render_performance_summary(perf: Dict[str, Any]) -> str:
    """Renders a concise performance timing summary."""
    lines = [
        "PERFORMANCE SUMMARY",
        "-" * 50,
        f"  Telemetry collection: {perf.get('capture_duration_seconds', 30.0):.1f}s",
        f"  Feature extraction:   {perf.get('feature_extraction_ms', 0.0):.1f} ms",
        f"  ML inference:         {perf.get('inference_ms', 0.0):.1f} ms",
        f"  Evidence analysis:    {perf.get('evidence_ms', 0.0):.1f} ms",
        f"  Discovery analysis:   {perf.get('discovery_ms', 0.0):.1f} ms",
        f"  Reporting latency:    {perf.get('reporting_ms', 0.0):.1f} ms",
        f"  Total analysis time:  {perf.get('total_analysis_ms', 0.0):.1f} ms",
        "-" * 50,
    ]
    return "\n".join(lines)


def render_diagnosis_report(result: Dict[str, Any], technical: bool = False) -> str:
    """Renders the comprehensive CURIO diagnosis report.

    Args:
        result: The structured dictionary returned by CurioDiagnosisPipeline.
        technical: If True, appends full technical telemetry tables, 11-window
                   probabilities, Kendall tau metrics, and performance timings.

    Returns:
        Formatted human-readable string.
    """
    lines: List[str] = []
    lines.append("=" * 50)
    lines.append("CURIO DIAGNOSIS")
    lines.append("=" * 50)
    lines.append("")

    diag = result.get("diagnosis", {})
    abn = result.get("abnormality", {})
    ev = result.get("evidence", {})
    disc = result.get("discovery", {})
    perf = result.get("performance", {})

    cond_raw = diag.get("condition", "normal")
    is_normal = (cond_raw == "normal")
    session_abnormal = abn.get("session_abnormal", False)

    # 1. Operating Status & Condition
    status_str = "ABNORMAL" if session_abnormal else "NORMAL"
    lines.append("Status:")
    lines.append(status_str)
    lines.append("")

    cond_display = "Normal Operation" if is_normal else cond_raw.replace("_", " ").title()
    lines.append("Condition:")
    lines.append(cond_display)
    lines.append("")

    # 2. Confidence & Abnormality Score
    confidence = diag.get("confidence", 0.0)
    lines.append("Model confidence:")
    lines.append(f"{confidence * 100:.0f}%")
    lines.append("")

    abn_score = abn.get("mean_score", 0.0)
    lines.append("Abnormality score:")
    lines.append(f"{abn_score:.2f}")
    lines.append("")

    # 3. Evidence Section (Why CURIO thinks this)
    if is_normal:
        lines.append("WHY?")
        lines.append("")
        lines.append("Telemetry remained consistent with the learned Normal operating range.")
        lines.append("")
    else:
        lines.append("WHY CURIO THINKS THIS")
        lines.append("")
        sup = ev.get("supporting_evidence", [])
        if sup:
            for idx, item in enumerate(sup, 1):
                stmt = item.get("human_readable_statement") or item.get("statement", "")
                lines.append(f"{idx}. {stmt}")
        else:
            lines.append("1. Telemetry pattern matched learned abnormal profile.")
        lines.append("")

    # 4. Discovery Section
    lines.append("CURIO DISCOVERY")
    lines.append("")

    disc_status = disc.get("discovery_status", "OPERATING_EQUILIBRIUM")
    if is_normal or disc_status == "OPERATING_EQUILIBRIUM":
        lines.append("Operating Equilibrium:")
        lines.append("No escalation detected.")
        lines.append("")
    else:
        obs_seq = disc.get("observed_sequence", [])
        if obs_seq:
            lines.append("Observed progression:")
            lines.append("")
            for idx, feat in enumerate(obs_seq):
                ev_info = next((e for e in disc.get("onset_events", []) if e.get("feature_name") == feat), None)
                t_str = f" (t = {ev_info['onset_time_seconds']:.1f}s)" if ev_info else ""
                lines.append(f"  {feat}{t_str}")
                if idx < len(obs_seq) - 1:
                    lines.append("    v")
            lines.append("")
        else:
            lines.append("Observed progression:")
            lines.append("None detected.")
            lines.append("")

        tau_norm = disc.get("normalized_tau")
        tau_raw = disc.get("kendall_tau")
        if tau_norm is not None and tau_raw is not None:
            lines.append("Temporal similarity:")
            lines.append(f"tau = {tau_raw:+.2f} (normalized: {tau_norm:.2f})")
            lines.append("")
        else:
            lines.append("Temporal similarity:")
            lines.append("N/A (sustained pressure or insufficient distinct onsets)")
            lines.append("")

        cov = disc.get("discovery_coverage", 0.0)
        lines.append("Discovery coverage:")
        lines.append(f"{cov * 100:.0f}%")
        lines.append("")

    # 5. Interpretation
    interp = disc.get("interpretation") or ev.get("overall_interpretation") or ev.get("interpretation") or ""
    if interp:
        lines.append("INTERPRETATION")
        lines.append("")
        lines.append(f'"{interp}"')
        lines.append("")

    lines.append("=" * 50)

    # 6. Technical Mode Appendices
    if technical:
        lines.append("")
        lines.append("=" * 60)
        lines.append("CURIO TECHNICAL DIAGNOSTIC DETAILS")
        lines.append("=" * 60)

        sess_id = result.get("session_id", "N/A")
        started = result.get("capture_started_at", "N/A")
        dur = result.get("capture_duration_seconds", 30.0)
        lines.append(f"Session ID:         {sess_id}")
        lines.append(f"Started At:         {started}")
        lines.append(f"Capture Duration:   {dur:.1f}s (61 samples @ 2 Hz)")
        lines.append("-" * 60)

        # Class Probabilities
        lines.append("Class Probabilities (Mean across 11 windows):")
        probs = diag.get("class_probabilities", {})
        for c, p in sorted(probs.items()):
            lines.append(f"  - {c:<20}: {p:.4f} ({p * 100:.1f}%)")
        lines.append("-" * 60)

        # Abnormality Metrics
        thresh = abn.get("threshold", 0.0)
        abn_cnt = abn.get("abnormal_window_count", 0)
        abn_ratio = abn.get("abnormal_window_ratio", 0.0)
        lines.append("Abnormality Assessment:")
        lines.append(f"  - Deployment Threshold:   {thresh:.4f}")
        lines.append(f"  - Mean Abnormality Score: {abn.get('mean_score', 0.0):.4f}")
        lines.append(f"  - Max Abnormality Score:  {abn.get('max_score', 0.0):.4f}")
        lines.append(f"  - Final Window Score:     {abn.get('final_window_score', 0.0):.4f}")
        lines.append(f"  - Abnormal Windows:       {abn_cnt}/11 ({abn_ratio * 100:.1f}%)")
        lines.append(f"  - Session Abnormal Flag:  {session_abnormal}")
        lines.append("-" * 60)

        # 11-Window Trajectory Table
        win_scores = abn.get("per_window_scores", [])
        win_preds = diag.get("per_window_predictions", [])
        if win_scores and win_preds:
            lines.append("11-Window Diagnostic Trajectory:")
            lines.append(f"  {'Win':<4} {'Time Range':<14} {'Condition':<18} {'P(Normal)':<10} {'Abnormal':<10} {'Status'}")
            for w_idx in range(len(win_scores)):
                w_pred = win_preds[w_idx] if w_idx < len(win_preds) else {}
                t_start = w_idx * 2.5
                t_end = t_start + 5.0
                t_str = f"{t_start:.1f}s-{t_end:.1f}s"
                w_cond = w_pred.get("predicted_condition", "N/A")
                p_norm = w_pred.get("probability_normal", 0.0)
                score = win_scores[w_idx]
                flag = "ABN" if score > thresh else "OK"
                lines.append(f"  {w_idx:<4} {t_str:<14} {w_cond:<18} {p_norm:<10.3f} {score:<10.3f} {flag}")
            lines.append("-" * 60)

        # Evidence Feature Deviations (Representative Window)
        rep_idx = ev.get("representative_window_idx", "N/A")
        lines.append(f"Evidence Feature Deviations (Window {rep_idx}):")
        lines.append(f"  {'Feature':<22} {'Observed':<10} {'Normal Ref':<12} {'z-dir':<8} {'Status'}")
        all_eval = ev.get("all_evaluated_features")
        if not all_eval:
            all_eval = ev.get("supporting_evidence", []) + ev.get("neutral_features", []) + ev.get("contradictory_evidence", [])
        for f_item in all_eval:
            fname = f_item.get("feature_name", "")
            obs_v = f_item.get("observed_value", 0.0)
            ref_v = f_item.get("normal_reference", f_item.get("reference_mean", 0.0))
            z_dir = f_item.get("directional_score", f_item.get("directional_deviation", 0.0))
            status = f_item.get("evidence_strength", f_item.get("status", "neutral")).upper()
            lines.append(f"  {fname:<22} {obs_v:<10.2f} {ref_v:<12.2f} {z_dir:<+8.2f} {status}")
        lines.append("-" * 60)

        # Discovery Details
        lines.append("Discovery Trajectory Details:")
        lines.append(f"  - Discovery Status:     {disc_status}")
        lines.append(f"  - Canonical Sequence:   {', '.join(disc.get('canonical_sequence', [])) or 'None'}")
        lines.append(f"  - Observed Sequence:    {', '.join(disc.get('observed_sequence', [])) or 'None'}")
        lines.append(f"  - Discovery Coverage:   {disc.get('discovery_coverage', 0.0) * 100:.1f}%")
        if disc.get("missing_features"):
            lines.append(f"  - Missing Canonical:    {', '.join(disc['missing_features'])}")
        if disc.get("unexpected_features"):
            lines.append(f"  - Unexpected Observed:  {', '.join(disc['unexpected_features'])}")
        lines.append("-" * 60)

        # Performance Timings
        lines.append(render_performance_summary(perf))
        lines.append("=" * 60)

    report_content = "\n".join(lines)
    return validate_no_causal_language(report_content)
