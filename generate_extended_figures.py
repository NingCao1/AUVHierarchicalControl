"""generate_extended_figures.py — Generate the extended paper figures.

Produces eight additional figures beyond generate_figures.py:
  1. control_effort_comparison.pdf  — instantaneous torque norm + cumulative IST
  2. observer_convergence.pdf       — disturbance observer estimation error
  3. noise_robustness.pdf           — RMS error vs. IMU noise level
  4. freq_sensitivity.pdf           — RMS error vs. disturbance frequency
  5. extended_mc_cdf.pdf            — 200-run Monte Carlo CDF
  6. combined_scenario.pdf          — simultaneous step + disturbance
  7. sinusoidal_tracking.pdf        — sinusoidal reference under disturbance
  8. gain_sensitivity.pdf           — gain sweep for Λ, Ks, L

Usage:
    python generate_extended_figures.py

All outputs are written to the same directory as this script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from simulation import (
    simulate,
    rms_error_deg, max_error_deg, settling_time,
    integrated_squared_torque, control_variation,
    run_wilcoxon_tests,
    TIME, DEG, T_FILTER_DEFAULT,
    MONTE_CARLO_CONTROLLERS,
)

# ── Plotting style ─────────────────────────────────────────────────────────────
np.random.seed(42)

rcParams["font.family"]     = "serif"
rcParams["font.serif"]      = ["Times New Roman"]
rcParams["font.size"]       = 10
rcParams["axes.labelsize"]  = 10
rcParams["axes.titlesize"]  = 11
rcParams["xtick.labelsize"] = 9
rcParams["ytick.labelsize"] = 9
rcParams["legend.fontsize"] = 9
rcParams["figure.dpi"]      = 300
rcParams["savefig.dpi"]     = 300
rcParams["savefig.format"]  = "pdf"
rcParams["savefig.bbox"]    = "tight"
rcParams["savefig.pad_inches"] = 0.1

ROOT = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Control Effort Comparison
# ══════════════════════════════════════════════════════════════════════════════
def generate_control_effort_figure() -> None:
    print("  [1/8] Control effort comparison …")
    ctrls = ["proposed", "adrc", "smc", "lqr", "pid"]
    labels = {"proposed": "Proposed", "adrc": "ADRC",
              "smc": "SMC", "lqr": "LQR", "pid": "PID"}

    results = {c: simulate(c, "disturbance", rng=np.random.default_rng(42))
               for c in ctrls}

    print("    Metrics:")
    for c in ctrls:
        ist  = integrated_squared_torque(results[c]["tau"])
        peak = float(np.max(np.linalg.norm(results[c]["tau"], axis=1)))
        cv   = control_variation(results[c]["u"])
        print(f"    {labels[c]:10s}  IST={ist:.2f}  Peak={peak:.2f}  MeanDu={cv:.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.5), sharex=True)
    styles = {
        "proposed": ("Proposed", "b-",  1.8),
        "adrc":     ("ADRC",     "c-",  1.5),
        "smc":      ("SMC",      "m--", 1.5),
    }
    for ctrl, (label, style, lw) in styles.items():
        tau_norm = np.linalg.norm(results[ctrl]["tau"], axis=1)
        axes[0].plot(TIME, tau_norm, style, linewidth=lw, label=label)
    axes[0].axvline(8.0, color="gray", linestyle=":", linewidth=1.0, label="Dist. onset")
    axes[0].set_ylabel("Torque norm (N$\\cdot$m)")
    axes[0].set_title("(a) Instantaneous torque norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=2, frameon=True)

    for ctrl, (label, style, lw) in styles.items():
        cum_ist = np.cumsum(np.sum(results[ctrl]["tau"] ** 2, axis=1)) * (TIME[1] - TIME[0])
        axes[1].plot(TIME, cum_ist, style, linewidth=lw, label=label)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Cumulative IST (N$^2$m$^2$s)")
    axes[1].set_title("(b) Integrated squared torque")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(ROOT / "control_effort_comparison.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Observer Convergence
# ══════════════════════════════════════════════════════════════════════════════
def generate_observer_convergence_figure() -> None:
    print("  [2/8] Observer convergence …")
    result = simulate("proposed", "disturbance", rng=np.random.default_rng(42))

    d_hat       = result["d_hat"]
    d_true      = result["d_true"]
    d_error     = np.linalg.norm(d_true - d_hat, axis=1)
    d_true_norm = np.linalg.norm(d_true, axis=1)

    idx_3, idx_8, idx_10 = (np.searchsorted(TIME, t) for t in [3.0, 8.0, 10.0])
    ss_before = float(np.sqrt(np.mean(d_error[idx_3:idx_8] ** 2)))
    peak_on   = float(np.max(d_error[idx_8:idx_8 + 100]))
    ss_after  = float(np.sqrt(np.mean(d_error[idx_10:] ** 2)))
    d_rms     = float(np.sqrt(np.mean(d_true_norm[idx_10:] ** 2)))
    ratio     = ss_after / d_rms * 100 if d_rms > 0 else 0.0

    print(f"    SS RMS before disturbance : {ss_before:.4f}")
    print(f"    Peak error at onset       : {peak_on:.4f}")
    print(f"    SS RMS after disturbance  : {ss_after:.4f}")
    print(f"    Estimation-to-disturbance : {ratio:.1f}%")

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True)

    axes[0].plot(TIME, d_error, "b-", linewidth=1.2,
                 label="$\\|\\tilde{\\mathbf{d}}\\|$")
    axes[0].axvline(8.0, color="red", linestyle="--", linewidth=1.0, alpha=0.7,
                    label="Dist. onset")
    axes[0].set_ylabel("Observer error norm")
    axes[0].set_title("(a) Disturbance observer estimation error")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", frameon=True)

    axes[1].plot(TIME, d_true_norm, "r-",  linewidth=1.2, alpha=0.7,
                 label="$\\|\\mathbf{d}\\|$ (true)")
    axes[1].plot(TIME, np.linalg.norm(d_hat, axis=1), "b--", linewidth=1.2,
                 label="$\\|\\hat{\\mathbf{d}}\\|$ (estimate)")
    axes[1].axvline(8.0, color="gray", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Disturbance acceleration norm")
    axes[1].set_title("(b) True vs. estimated disturbance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(ROOT / "observer_convergence.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: IMU Noise Robustness
# ══════════════════════════════════════════════════════════════════════════════
def generate_noise_robustness_figure() -> None:
    print("  [3/8] IMU noise robustness …")
    sigma_values = [0.005, 0.01, 0.02, 0.03, 0.05]
    n_trials     = 10
    dist_start   = np.searchsorted(TIME, 8.0)
    ctrls        = ["proposed", "adrc", "smc"]

    raw: dict[str, dict] = {c: {s: [] for s in sigma_values} for c in ctrls}

    for sig in sigma_values:
        for trial in range(n_trials):
            for ctrl in ctrls:
                r = simulate(ctrl, "disturbance",
                             sigma_omega=sig,
                             rng=np.random.default_rng(1000 + trial))
                raw[ctrl][sig].append(
                    rms_error_deg(r["eta"], r["eta_ref"], dist_start)
                )

    means = {c: [np.mean(raw[c][s]) for s in sigma_values] for c in ctrls}
    stds  = {c: [np.std(raw[c][s])  for s in sigma_values] for c in ctrls}

    print("    Mean RMS per noise level:")
    for ctrl in ctrls:
        print(f"    {ctrl:10s}: {[f'{m:.4f}' for m in means[ctrl]]}")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    styles = {"proposed": ("Proposed", "b-o"),
              "adrc":     ("ADRC",     "c-s"),
              "smc":      ("SMC",      "m--^")}
    for ctrl, (label, style) in styles.items():
        ax.errorbar(sigma_values, means[ctrl], yerr=stds[ctrl],
                    fmt=style, linewidth=1.5, markersize=6, capsize=4, label=label)
    ax.set_xlabel("IMU noise std. $\\sigma_\\omega$ (rad/s)")
    ax.set_ylabel("Disturbance-case RMS error (deg)")
    ax.set_title("Noise robustness: RMS error vs. IMU noise level")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(ROOT / "noise_robustness.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Disturbance Frequency Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
def generate_freq_sensitivity_figure() -> None:
    print("  [4/8] Disturbance frequency sensitivity …")
    freq_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    dist_start  = np.searchsorted(TIME, 8.0)
    ctrls       = ["proposed", "adrc", "smc"]

    rms_results: dict[str, list[float]] = {c: [] for c in ctrls}

    for freq in freq_values:
        for ctrl in ctrls:
            r = simulate(ctrl, "disturbance",
                         freq_scale=freq,
                         rng=np.random.default_rng(42))
            rms_results[ctrl].append(
                rms_error_deg(r["eta"], r["eta_ref"], dist_start)
            )

    print("    RMS per frequency:")
    for ctrl in ctrls:
        print(f"    {ctrl:10s}: {[f'{v:.4f}' for v in rms_results[ctrl]]}")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    styles = {"proposed": ("Proposed", "b-o"),
              "adrc":     ("ADRC",     "c-s"),
              "smc":      ("SMC",      "m--^")}
    for ctrl, (label, style) in styles.items():
        ax.plot(freq_values, rms_results[ctrl], style,
                linewidth=1.5, markersize=6, label=label)
    ax.set_xlabel("Disturbance base frequency $\\omega_d$ (rad/s)")
    ax.set_ylabel("Disturbance-case RMS error (deg)")
    ax.set_title("Disturbance frequency sensitivity")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(ROOT / "freq_sensitivity.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Extended Monte Carlo CDF (200 runs)
# ══════════════════════════════════════════════════════════════════════════════
def generate_extended_mc_figure() -> dict[str, list[float]]:
    print("  [5/8] Extended Monte Carlo (200 runs) …")
    dist_start = np.searchsorted(TIME, 8.0)
    n_runs     = 200

    mc_runs: dict[str, list[float]] = {c: [] for c in MONTE_CARLO_CONTROLLERS}

    for seed in range(100, 100 + n_runs):
        trial_rng   = np.random.default_rng(seed)
        initial_eta = np.deg2rad(
            trial_rng.uniform(low=[-3.0, -2.0, -3.0], high=[3.0, 2.0, 3.0])
        )
        mismatch_pct = float(trial_rng.uniform(0.05, 0.20))
        tf           = float(trial_rng.choice([0.02, 0.05, 0.10, 0.20]))

        for i, ctrl in enumerate(MONTE_CARLO_CONTROLLERS):
            r = simulate(ctrl, "disturbance",
                         tf=tf, mismatch_pct=mismatch_pct,
                         scenario="nominal",
                         rng=np.random.default_rng(seed * 100 + i),
                         initial_eta=initial_eta)
            mc_runs[ctrl].append(
                rms_error_deg(r["eta"], r["eta_ref"], dist_start)
            )

    print("    Extended MC stats (N=200):")
    for ctrl in MONTE_CARLO_CONTROLLERS:
        arr = np.asarray(mc_runs[ctrl])
        print(f"    {ctrl:10s}: mean={arr.mean():.4f} ± {arr.std():.4f}  "
              f"med={np.median(arr):.4f}  p95={np.percentile(arr,95):.4f}  "
              f"p99={np.percentile(arr,99):.4f}")

    print("\n    Wilcoxon tests (proposed vs each baseline):")
    wilcoxon = run_wilcoxon_tests(mc_runs)
    for key, val in wilcoxon.items():
        print(f"    {key}: p={val['p_value']:.2e},  effect_r={val['effect_r']:.3f}")

    # CDF plot
    colors = {"proposed": "#4c72b0", "adrc": "#64b5cd", "smc": "#dd8452",
              "lqr": "#55a868", "pid": "#c44e52"}
    labels = {"proposed": "Proposed", "adrc": "ADRC", "smc": "SMC",
              "lqr": "LQR", "pid": "PID"}

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for ctrl in MONTE_CARLO_CONTROLLERS:
        sv  = np.sort(mc_runs[ctrl])
        cdf = np.arange(1, len(sv) + 1) / len(sv)
        ax.plot(sv, cdf, linewidth=1.8, color=colors[ctrl], label=labels[ctrl])
    ax.set_xlabel("Disturbance-case RMS attitude error (deg)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"CDF of RMS error over {n_runs} Monte Carlo runs")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="lower right")
    ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(ROOT / "extended_mc_cdf.pdf")
    plt.close(fig)

    return mc_runs


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6: Combined Tracking + Disturbance Scenario
# ══════════════════════════════════════════════════════════════════════════════
def generate_combined_scenario_figure() -> None:
    print("  [6/8] Combined tracking + disturbance scenario …")
    ctrls = ["proposed", "adrc", "smc", "lqr", "pid"]
    labels = {"proposed": "Proposed", "adrc": "ADRC",
              "smc": "SMC", "lqr": "LQR", "pid": "PID"}

    results = {c: simulate(c, "combined",
                           dist_onset=4.0,
                           rng=np.random.default_rng(42))
               for c in ctrls}

    step_start  = np.searchsorted(TIME, 2.0)
    dist_start  = np.searchsorted(TIME, 4.0)
    target_roll = np.deg2rad(20.0)

    print("    Metrics:")
    for ctrl in ctrls:
        track_rms = rms_error_deg(results[ctrl]["eta"], results[ctrl]["eta_ref"], step_start)
        dist_rms  = rms_error_deg(results[ctrl]["eta"], results[ctrl]["eta_ref"], dist_start)
        max_err   = max_error_deg(results[ctrl]["eta"], results[ctrl]["eta_ref"], step_start)
        settle    = settling_time(TIME, results[ctrl]["eta"][:, 0], target_roll, 2.0)
        print(f"    {labels[ctrl]:10s}: settle={settle:.2f}s  "
              f"track_rms={track_rms:.3f}  dist_rms={dist_rms:.3f}  max={max_err:.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.5), sharex=True)
    ax_names   = ["(a) Roll", "(b) Pitch", "(c) Yaw"]
    plot_ctrls = {"proposed": ("Proposed", "b-"),
                  "adrc":     ("ADRC",     "c-"),
                  "smc":      ("SMC",      "m--")}

    for i, ax_name in enumerate(ax_names):
        ax = axes[i]
        ax.plot(TIME, results["proposed"]["eta_ref"][:, i] * DEG,
                "k--", linewidth=1.2, label="Reference")
        for ctrl, (label, style) in plot_ctrls.items():
            ax.plot(TIME, results[ctrl]["eta"][:, i] * DEG,
                    style, linewidth=1.5, label=label)
        ax.axvline(2.0, color="green", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(4.0, color="red",   linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(f"{ax_name} (deg)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower right", ncol=2, frameon=True, fontsize=8)
            ax.annotate("Step",  xy=(2.05, ax.get_ylim()[0]), fontsize=7, color="green")
            ax.annotate("Dist.", xy=(4.05, ax.get_ylim()[0]), fontsize=7, color="red")
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig(ROOT / "combined_scenario.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7: Sinusoidal Trajectory Tracking
# ══════════════════════════════════════════════════════════════════════════════
def generate_sinusoidal_tracking_figure() -> None:
    print("  [7/8] Sinusoidal trajectory tracking …")
    ctrls  = ["proposed", "adrc", "smc", "lqr", "pid"]
    labels = {"proposed": "Proposed", "adrc": "ADRC",
              "smc": "SMC", "lqr": "LQR", "pid": "PID"}

    # disturbance active from t=0 (dist_onset=0)
    results = {c: simulate(c, "sinusoidal_dist",
                           dist_onset=0.0,
                           rng=np.random.default_rng(42))
               for c in ctrls}

    ss_start = np.searchsorted(TIME, 5.0)
    print("    Tracking metrics (t > 5 s):")
    for ctrl in ctrls:
        rms  = rms_error_deg(results[ctrl]["eta"], results[ctrl]["eta_ref"], ss_start)
        peak = max_error_deg(results[ctrl]["eta"], results[ctrl]["eta_ref"], ss_start)
        print(f"    {labels[ctrl]:10s}: RMS={rms:.4f}  Peak={peak:.4f}")

    fig, axes = plt.subplots(4, 1, figsize=(6.5, 8.0), sharex=True)
    ax_names   = ["(a) Roll", "(b) Pitch", "(c) Yaw"]
    plot_ctrls = {"proposed": ("Proposed", "b-"),
                  "adrc":     ("ADRC",     "c-"),
                  "smc":      ("SMC",      "m--")}

    for i, ax_name in enumerate(ax_names):
        ax = axes[i]
        ax.plot(TIME, results["proposed"]["eta_ref"][:, i] * DEG,
                "k--", linewidth=1.4, label="Reference")
        for ctrl, (label, style) in plot_ctrls.items():
            ax.plot(TIME, results[ctrl]["eta"][:, i] * DEG,
                    style, linewidth=1.2, label=label)
        ax.set_ylabel(f"{ax_name} (deg)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", ncol=2, frameon=True, fontsize=8)

    for ctrl, (label, style) in plot_ctrls.items():
        err_norm = np.linalg.norm(
            (results[ctrl]["eta"] - results[ctrl]["eta_ref"]) * DEG, axis=1
        )
        axes[3].plot(TIME, err_norm, style, linewidth=1.2, label=label)
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("(d) Error norm (deg)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right", frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(ROOT / "sinusoidal_tracking.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8: Controller Gain Sensitivity
# ══════════════════════════════════════════════════════════════════════════════
def generate_gain_sensitivity_figure() -> None:
    print("  [8/8] Gain sensitivity sweep …")
    kappa_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    dist_start   = np.searchsorted(TIME, 8.0)
    step_start   = np.searchsorted(TIME, 2.0)
    target_roll  = np.deg2rad(20.0)

    gain_names = ["$\\mathbf{\\Lambda}$", "$\\mathbf{K}_s$", "$\\mathbf{L}$"]
    gain_keys  = ["lambda_scale", "ks_scale", "obs_scale"]

    all_results: dict[str, dict] = {}

    for gi, gkey in enumerate(gain_keys):
        rms_v, settle_v, over_v = [], [], []
        for kappa in kappa_values:
            gs = {gkey: kappa}

            r_dist = simulate("proposed", "disturbance",
                              gain_scale=gs, rng=np.random.default_rng(42))
            rms_v.append(rms_error_deg(r_dist["eta"], r_dist["eta_ref"], dist_start))

            r_step = simulate("proposed", "step",
                              gain_scale=gs, rng=np.random.default_rng(42))
            settle_v.append(settling_time(TIME, r_step["eta"][:, 0], target_roll, 2.0))
            peak = float(np.max(r_step["eta"][step_start:, 0]))
            over_v.append(max(0.0, (peak - target_roll) / abs(target_roll) * 100.0))

        all_results[gkey] = {"rms": rms_v, "settle": settle_v, "overshoot": over_v}

        print(f"    {gain_names[gi]}:")
        print(f"      RMS:      {[f'{v:.4f}' for v in rms_v]}")
        print(f"      Settle:   {[f'{v:.2f}' for v in settle_v]}")
        print(f"      Overshoot:{[f'{v:.2f}' for v in over_v]}")

    fig, axes = plt.subplots(3, 3, figsize=(10.0, 8.0))
    metric_names = ["Dist. RMS (deg)", "Settling time (s)", "Overshoot (%)"]
    metric_keys  = ["rms", "settle", "overshoot"]
    colors       = ["#4c72b0", "#dd8452", "#55a868"]

    for gi, gkey in enumerate(gain_keys):
        for mi, mkey in enumerate(metric_keys):
            ax = axes[gi][mi]
            ax.plot(kappa_values, all_results[gkey][mkey],
                    "o-", color=colors[mi], linewidth=1.5, markersize=6)
            ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.grid(True, alpha=0.3)
            if gi == 0:
                ax.set_title(metric_names[mi], fontsize=10)
            if gi == 2:
                ax.set_xlabel("Scale factor $\\kappa$")
            if mi == 0:
                ax.set_ylabel(gain_names[gi], fontsize=11)

    plt.tight_layout()
    plt.savefig(ROOT / "gain_sensitivity.pdf")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("=" * 60)
    print("Generating extended figures for the AUV paper")
    print("=" * 60)

    generate_control_effort_figure()
    generate_observer_convergence_figure()
    generate_noise_robustness_figure()
    generate_freq_sensitivity_figure()
    generate_extended_mc_figure()
    generate_combined_scenario_figure()
    generate_sinusoidal_tracking_figure()
    generate_gain_sensitivity_figure()

    print("\n" + "=" * 60)
    print("All 8 extended figures saved to:", ROOT)
    print("=" * 60)


if __name__ == "__main__":
    main()
