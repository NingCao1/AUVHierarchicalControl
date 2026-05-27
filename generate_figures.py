"""generate_figures.py — Generate the main paper figures and numerical tables.

Produces:
  attitude_tracking.pdf
  disturbance_rejection.pdf
  fin_coordination.pdf
  allocation_ablation.png
  sensitivity_analysis.png
  experimental_results.png
  numerical_metrics.json
  sensitivity_metrics.json

Usage:
    python generate_figures.py

All outputs are written to the same directory as this script.
Random seed is fixed (numpy seed 42) for full reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from simulation import (
    simulate, compute_metrics,
    run_sensitivity, run_generalization,
    TIME, DEG, T_FILTER_DEFAULT,
    MONTE_CARLO_CONTROLLERS,
)
from auv_model import GENERALIZATION_SCENARIOS, SCENARIOS

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

CTRL_STYLES = {
    "proposed": ("Proposed", "b-",  1.8),
    "adrc":     ("ADRC",     "c-",  1.5),
    "smc":      ("SMC",      "m--", 1.5),
    "lqr":      ("LQR",      "g-.", 1.3),
    "pid":      ("PID",      "r:",  1.3),
}


# ── Figure 1: Multi-axis step response ────────────────────────────────────────
def plot_step(results: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.0), sharex=True)
    axis_names = ["Roll", "Pitch", "Yaw"]

    for ax_idx, ax_name in enumerate(axis_names):
        ax = axes[ax_idx]
        ax.plot(TIME, results["proposed"]["eta_ref"][:, ax_idx] * DEG,
                "k--", linewidth=1.4, label="Reference")
        for key, (label, style, lw) in CTRL_STYLES.items():
            ax.plot(TIME, results[key]["eta"][:, ax_idx] * DEG,
                    style, linewidth=lw, label=label)
        ax.set_ylabel(f"{ax_name} (deg)")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc="lower right", ncol=3, frameon=True)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(ROOT / "attitude_tracking.pdf")
    plt.close(fig)


# ── Figure 2: Disturbance rejection ───────────────────────────────────────────
def plot_disturbance(results: dict[str, dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.2), sharex=True)

    for key, (label, style, lw) in CTRL_STYLES.items():
        err_norm = np.linalg.norm(
            (results[key]["eta"] - results[key]["eta_ref"]) * DEG, axis=1
        )
        axes[0].plot(TIME, err_norm, style, linewidth=lw, label=label)

    axes[0].axvline(8.0, color="gray", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("Attitude error norm (deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=2, frameon=True)

    dist_norm = np.linalg.norm(results["proposed"]["tau_dist"], axis=1)
    axes[1].plot(TIME, dist_norm, "k-", linewidth=1.5, label="Injected disturbance")
    axes[1].axvline(8.0, color="gray", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Disturbance torque norm (N·m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(ROOT / "disturbance_rejection.pdf")
    plt.close(fig)


# ── Figure 3: Fin coordination ─────────────────────────────────────────────────
def plot_fin_coordination(proposed_step: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 5.0), sharex=True)

    for i in range(4):
        axes[0].plot(TIME, proposed_step["u"][:, i], linewidth=1.4, label=f"Fin {i+1}")
    axes[0].set_ylabel("Normalised fin command")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=2, frameon=True)

    tc = proposed_step["tau"]
    axes[1].plot(TIME, tc[:, 0], "b-",  linewidth=1.4, label="Roll torque")
    axes[1].plot(TIME, tc[:, 1], "r--", linewidth=1.4, label="Pitch torque")
    axes[1].plot(TIME, tc[:, 2], "g-.", linewidth=1.4, label="Yaw torque")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Allocated torque (N·m)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", ncol=2, frameon=True)

    plt.tight_layout()
    plt.savefig(ROOT / "fin_coordination.pdf")
    plt.close(fig)


# ── Figure 4: Ablation bar chart ───────────────────────────────────────────────
def plot_ablation(
    step_metrics:        dict[str, dict[str, float]],
    disturbance_metrics: dict[str, dict[str, float]],
) -> None:
    keys   = ["proposed", "proposed_no_observer", "proposed_no_smooth"]
    labels = ["Proposed", "No observer", "No smoothness"]

    rms          = [disturbance_metrics[k]["dist_rms_deg"]       for k in keys]
    overshoot_v  = [step_metrics[k]["step_overshoot_pct"]        for k in keys]
    peak_rate    = [step_metrics[k]["peak_cmd_rate"] * 100        for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.1))
    panels = [
        ("(a) Disturbance RMS", rms,         "Dist. RMS (deg)",    "#4c72b0"),
        ("(b) Step overshoot",  overshoot_v,  "Overshoot (%)",      "#dd8452"),
        ("(c) Peak cmd rate",   peak_rate,    "Peak cmd rate ×100", "#55a868"),
    ]

    for ax, (title, values, ylabel, color) in zip(axes, panels):
        bars  = ax.bar(labels, values, color=color, width=0.65)
        y_max = max(values) * 1.22 if max(values) > 0 else 1.0
        ax.set_ylim(0.0, y_max)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=16)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 0.025 * y_max,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(ROOT / "allocation_ablation.png")
    plt.close(fig)


# ── Figure 5: Sensitivity bar charts ──────────────────────────────────────────
def plot_sensitivity(sens: dict) -> None:
    tf_keys = list(sens["tf_sweep"].keys())
    tf_prop = [sens["tf_sweep"][k]["proposed_dist_rms"] for k in tf_keys]
    tf_smc  = [sens["tf_sweep"][k]["smc_dist_rms"]      for k in tf_keys]

    delta_keys = list(sens["delta_sweep"].keys())
    d_prop = [sens["delta_sweep"][k]["proposed_dist_rms"] for k in delta_keys]
    d_smc  = [sens["delta_sweep"][k]["smc_dist_rms"]      for k in delta_keys]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    x1 = np.arange(len(tf_keys))
    axes[0].bar(x1 - 0.2, tf_prop, 0.35, color="#4c72b0", label="Proposed")
    axes[0].bar(x1 + 0.2, tf_smc,  0.35, color="#dd8452", label="SMC")
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels([f"$T_f$={k} s" for k in tf_keys], fontsize=8)
    axes[0].set_ylabel("Dist. RMS error (deg)")
    axes[0].set_title("(a) Observer filter bandwidth sweep")
    axes[0].set_ylim(0.0, max(max(tf_prop), max(tf_smc)) * 1.22)
    axes[0].legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    axes[0].grid(True, axis="y", alpha=0.3)

    x2 = np.arange(len(delta_keys))
    axes[1].bar(x2 - 0.2, d_prop, 0.35, color="#4c72b0", label="Proposed")
    axes[1].bar(x2 + 0.2, d_smc,  0.35, color="#dd8452", label="SMC")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(["0%", "5%", "10%", "20%"], fontsize=8)
    axes[1].set_xlabel("Allocation mismatch $\\delta$")
    axes[1].set_ylabel("Dist. RMS error (deg)")
    axes[1].set_title("(b) Allocation mismatch sweep")
    axes[1].set_ylim(0.0, max(max(d_prop), max(d_smc)) * 1.22)
    axes[1].legend(frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(ROOT / "sensitivity_analysis.png")
    plt.close(fig)


# ── Figure 6: Generalization (Monte Carlo + scenario bars) ─────────────────────
def plot_generalization(generalization: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))

    # (a) Monte Carlo box plot
    keys   = ["proposed", "adrc", "smc", "lqr", "pid"]
    labels = ["Proposed", "ADRC", "SMC", "LQR", "PID"]
    colors = ["#4c72b0", "#64b5cd", "#dd8452", "#55a868", "#c44e52"]
    box_data = [generalization["monte_carlo_runs"][k] for k in keys]
    bplot = axes[0].boxplot(box_data, patch_artist=True,
                            tick_labels=labels, showfliers=False)
    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    axes[0].set_title("(a) Monte Carlo robustness")
    axes[0].set_ylabel("Disturbance-case RMS error (deg)")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].tick_params(axis="x", rotation=12)

    # (b) Scenario-wise bars
    scen_keys   = GENERALIZATION_SCENARIOS
    scen_labels = ["S1", "S2", "S3", "S4"]
    x     = np.arange(len(scen_keys))
    width = 0.24
    series = [
        ("proposed", "Proposed", "#4c72b0", -width),
        ("adrc",     "ADRC",     "#64b5cd",  0.0),
        ("smc",      "SMC",      "#dd8452",  width),
    ]
    for key, label, color, offset in series:
        vals = [generalization["scenario_results"][s][key] for s in scen_keys]
        axes[1].bar(x + offset, vals, width=width, label=label, color=color)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scen_labels, rotation=10)
    axes[1].set_title("(b) Multi-scenario generalization")
    axes[1].set_ylabel("Disturbance-case RMS error (deg)")
    top = max(generalization["scenario_results"][s]["smc"] for s in scen_keys)
    axes[1].set_ylim(0.0, top * 1.24)
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(frameon=True, loc="upper center",
                   bbox_to_anchor=(0.5, 1.18), ncol=3)

    plt.tight_layout()
    plt.savefig(ROOT / "experimental_results.png")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    all_controllers = [
        "proposed", "adrc", "smc", "lqr", "pid",
        "proposed_no_observer", "proposed_no_smooth",
    ]

    print("Running step-case simulations …")
    step_results = {
        name: simulate(name, "step", rng=np.random.default_rng(42))
        for name in all_controllers
    }

    print("Running disturbance-case simulations …")
    dist_results = {
        name: simulate(name, "disturbance", rng=np.random.default_rng(42))
        for name in all_controllers
    }

    step_metrics = compute_metrics(step_results)
    dist_metrics = compute_metrics(dist_results)

    print("Running sensitivity study …")
    sens = run_sensitivity()

    print("Running generalization study (40-run MC) …")
    generalization = run_generalization(n_mc=40)

    print("Generating figures …")
    plot_step(step_results)
    plot_disturbance(dist_results)
    plot_fin_coordination(step_results["proposed"])
    plot_ablation(step_metrics, dist_metrics)
    plot_sensitivity(sens)
    plot_generalization(generalization)

    # ── Save metrics ───────────────────────────────────────────────────────────
    summary = {
        "step":          step_metrics,
        "disturbance":   dist_metrics,
        "generalization": generalization,
    }
    (ROOT / "numerical_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (ROOT / "sensitivity_metrics.json").write_text(
        json.dumps(sens, indent=2), encoding="utf-8"
    )

    # ── Console report ─────────────────────────────────────────────────────────
    print("\n=== Disturbance-case RMS error (deg) ===")
    for name in ["proposed", "adrc", "smc", "lqr", "pid"]:
        rms = dist_metrics[name]["dist_rms_deg"]
        mx  = dist_metrics[name]["dist_max_deg"]
        print(f"  {name:25s}  RMS={rms:.4f}  Max={mx:.4f}")

    print("\n=== Step-case metrics ===")
    for name in ["proposed", "adrc", "smc", "lqr", "pid"]:
        st  = step_metrics[name]["step_settling_s"]
        ovs = step_metrics[name]["step_overshoot_pct"]
        print(f"  {name:25s}  Settle={st:.2f} s  Overshoot={ovs:.2f}%")

    print("\n=== Ablation (disturbance case) ===")
    for name in ["proposed", "proposed_no_observer", "proposed_no_smooth"]:
        rms = dist_metrics[name]["dist_rms_deg"]
        pcr = step_metrics[name]["peak_cmd_rate"]
        print(f"  {name:30s}  dist_rms={rms:.4f}  peak_cmd_rate={pcr:.5f}")

    print("\n=== Monte Carlo mean ± std (deg) ===")
    for name in MONTE_CARLO_CONTROLLERS:
        st = generalization["monte_carlo_stats"][name]
        print(f"  {name:25s}  {st['mean']:.4f} ± {st['std']:.4f}  (p95={st['p95']:.4f})")

    print("\nSaved: attitude_tracking.pdf, disturbance_rejection.pdf,")
    print("       fin_coordination.pdf, allocation_ablation.png,")
    print("       sensitivity_analysis.png, experimental_results.png,")
    print("       numerical_metrics.json, sensitivity_metrics.json")


if __name__ == "__main__":
    main()
