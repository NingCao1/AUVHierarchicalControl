# AUV Hierarchical Control — Simulation Code

Reproducible simulation code for the paper:


---

## Repository structure

```
code/
├── auv_model.py               # Physical model: vehicle parameters, disturbance model,
│                              #   fin allocator, reference profiles
├── controllers.py             # Control algorithms: proposed SMC+observer, ADRC, SMC,
│                              #   LQR, PID, and ablation variants
├── simulation.py              # Simulation engine: closed-loop integrator, performance
│                              #   metrics, sensitivity study, Monte Carlo study
├── generate_figures.py        # Reproduce main paper figures (Figs. 2–6, Tables 1–7)
├── generate_extended_figures.py  # Reproduce extended figures (Figs. 7–14)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

The four Python modules have a clean dependency structure:

```
auv_model.py  ←──  controllers.py
     ↑                   ↑
     └──────── simulation.py ──────→  generate_figures.py
                                  └→  generate_extended_figures.py
```

---

## Requirements

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

Dependencies: `numpy >= 1.23`, `matplotlib >= 3.6`, `scipy >= 1.9`.

> **Note:** `scipy` is used only for Wilcoxon rank-sum tests in the extended Monte Carlo figure.
> If your environment has a scipy/numpy version conflict, all other figures will still run correctly.

---

## Quick start — reproduce all paper results

```bash
# Step 1: main figures + numerical tables
python generate_figures.py

# Step 2: extended figures (observer convergence, noise sweep, MC CDF, etc.)
python generate_extended_figures.py
```

Both scripts are self-contained. All outputs are written to the same directory.
Results are fully deterministic: `numpy.random.seed(42)` is set at the top of every entry point.

---

## Step-by-step guide

### 1. Main simulation results (`generate_figures.py`)

Running `python generate_figures.py` executes the following in order:

| Step | What runs | Output |
|------|-----------|--------|
| Step-case simulations | 7 controllers × step reference | — |
| Disturbance-case simulations | 7 controllers × confined-env disturbance | — |
| Sensitivity study | T_f sweep + mismatch δ sweep | `sensitivity_metrics.json` |
| Generalization study | 4 scenarios + 40-run Monte Carlo | part of `numerical_metrics.json` |
| Figure generation | 6 figures saved | see table below |
| Console report | key metrics printed to terminal | — |

**Output files:**

| File | Paper reference |
|------|-----------------|
| `attitude_tracking.pdf` | Fig. — Multi-axis step response |
| `disturbance_rejection.pdf` | Fig. — Disturbance rejection comparison |
| `fin_coordination.pdf` | Fig. — Fin command trajectories |
| `allocation_ablation.png` | Fig. — Ablation bar chart |
| `sensitivity_analysis.png` | Fig. — T_f and δ sensitivity |
| `experimental_results.png` | Fig. — Monte Carlo + scenario bars |
| `numerical_metrics.json` | All tabulated values (Tables 2–7) |
| `sensitivity_metrics.json` | Table 3 values |

Expected console output (key metrics):

```
=== Disturbance-case RMS error (deg) ===
  proposed                   RMS=0.0267  Max=0.0497
  adrc                       RMS=0.0440  Max=0.1207
  smc                        RMS=0.1077  Max=0.2301
  lqr                        RMS=1.6630  Max=3.3889
  pid                        RMS=3.7438  Max=7.7533

=== Step-case metrics ===
  proposed                   Settle=2.25 s  Overshoot=4.86%
  adrc                       Settle=1.49 s  Overshoot=0.91%
  smc                        Settle=2.75 s  Overshoot=0.54%
```

### 2. Extended figures (`generate_extended_figures.py`)

Running `python generate_extended_figures.py` produces 8 additional figures:

| Step | Output file | Paper reference |
|------|-------------|-----------------|
| [1/8] Control effort | `control_effort_comparison.pdf` | Fig. — IST + torque norm |
| [2/8] Observer convergence | `observer_convergence.pdf` | Fig. — d̂ vs d |
| [3/8] IMU noise robustness | `noise_robustness.pdf` | Table — σ_ω sweep |
| [4/8] Disturbance frequency | `freq_sensitivity.pdf` | Table — ω_d sweep |
| [5/8] Extended MC (200 runs) | `extended_mc_cdf.pdf` | Fig. — CDF + Wilcoxon |
| [6/8] Combined scenario | `combined_scenario.pdf` | Fig. — step + disturbance |
| [7/8] Sinusoidal tracking | `sinusoidal_tracking.pdf` | Table — sinusoidal RMS |
| [8/8] Gain sensitivity | `gain_sensitivity.pdf` | Table — Λ, K_s, L sweep |

This script takes longer (~10–20 min) because the 200-run Monte Carlo runs all 5 controllers per trial.

---

## Running a custom simulation

You can call `simulate()` directly from `simulation.py` for any custom experiment:

```python
from simulation import simulate, rms_error_deg, TIME
import numpy as np

# Single run: proposed controller, disturbance case, narrow-corridor scenario
result = simulate(
    controller  = "proposed",
    case        = "disturbance",
    scenario    = "narrow_corridor",   # S2
    mismatch_pct= 0.15,                # 15% allocation mismatch
    sigma_omega = 0.02,                # noisier IMU
    tf          = 0.05,                # observer filter time constant [s]
    rng         = np.random.default_rng(0),
)

dist_start = int(8.0 / 0.01)   # index for t = 8 s
print(f"RMS error: {rms_error_deg(result['eta'], result['eta_ref'], dist_start):.4f} deg")
```

**Available `case` values:**

| `case` | Description |
|--------|-------------|
| `"step"` | Multi-axis step command at t = 2 s |
| `"disturbance"` | Constant hold + boundary disturbance from t = 8 s |
| `"combined"` | Step at t = 2 s + disturbance from t = 4 s |
| `"sinusoidal_dist"` | Sinusoidal reference + disturbance throughout |

**Available `controller` values:** `"proposed"`, `"adrc"`, `"smc"`, `"lqr"`, `"pid"`, `"proposed_no_observer"`, `"proposed_no_smooth"`

**Simulation output dict keys:**

| Key | Shape | Description |
|-----|-------|-------------|
| `time` | (N,) | Time vector [s] |
| `eta` | (N, 3) | Euler angles [φ, θ, ψ] [rad] |
| `eta_ref` | (N, 3) | Reference attitude [rad] |
| `omega` | (N, 3) | Body angular velocity [rad/s] |
| `tau` | (N, 3) | True plant torque [N·m] |
| `tau_nom` | (N, 3) | Nominal torque G_nom @ u [N·m] |
| `tau_dist` | (N, 3) | External disturbance torque [N·m] |
| `u` | (N, 4) | Fin commands (normalised) |
| `d_hat` | (N, 3) | Observer estimate of lumped disturbance |
| `d_true` | (N, 3) | True lumped disturbance acceleration |
| `alloc_residual` | (N, 3) | τ_actual − τ_nom [N·m] |
| `surface` | (N, 3) | Sliding surface / tracking error |

---

## System model summary

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Roll inertia | I_xx | 2.5 kg·m² |
| Pitch / yaw inertia | I_yy = I_zz | 5.0 kg·m² |
| Number of fins | n_f | 4 |
| Fin command bounds | [u_min, u_max] | [−1.2, 1.2] |
| Actuator lag | τ_a | 0.08 s |
| Sampling interval | Δt | 0.01 s |
| Simulation duration | T | 20 s |
| IMU noise std (default) | σ_ω | 0.01 rad/s |
| Allocation mismatch (default) | δ | 10% |
| ADRC ESO bandwidth | ω_o | 12 rad/s |
| ADRC controller bandwidth | ω_c | 4.0 rad/s |

Confined-environment scenarios:

| Scenario | Wall d_w (m) | Ground d_g (m) | Peak disturbance |
|----------|-------------|----------------|-----------------|
| S1 Nominal corridor | 0.20–0.28 | 0.25–0.31 | 0.155 N·m |
| S2 Narrow corridor | 0.15–0.21 | 0.27–0.33 | 0.169 N·m |
| S3 Seabed-hover | 0.27–0.33 | 0.16–0.20 | 0.147 N·m |
| S4 Compound boundary | 0.16–0.22 | 0.18–0.22 | 0.216 N·m |

---
