"""auv_model.py — AUV physical model: parameters, dynamics, disturbance, allocation.

This module contains everything that belongs to the *plant* side:
  - Vehicle inertia and actuator parameters
  - Nominal / actual fin effectiveness matrices
  - Confined-environment disturbance model (wall + ground + residual + jet)
  - Regularised torque-to-fin allocator
  - Reference attitude profiles
"""

from __future__ import annotations

import numpy as np

# ── Vehicle parameters ─────────────────────────────────────────────────────────
INERTIA = np.diag([2.5, 5.0, 5.0])       # kg·m²  [Ixx, Iyy, Izz]
INERTIA_INV = np.linalg.inv(INERTIA)

# Nominal control-effectiveness matrix G ∈ R^{3×4}  (known to all controllers)
G_NOM = np.array([
    [ 1.00, -1.00,  0.90, -0.90],
    [ 0.75,  0.75, -0.75, -0.75],
    [ 0.60, -0.60, -0.60,  0.60],
])

# Default true plant matrix: ≈10 % row-wise gain mismatch + additive offset
G_ACTUAL_DEFAULT = G_NOM @ np.diag([1.00, 0.92, 1.05, 0.95]) + np.array([
    [ 0.03,  0.00, -0.02,  0.01],
    [ 0.00, -0.02,  0.01,  0.02],
    [ 0.02, -0.01,  0.00, -0.02],
])

# Fin command bounds (normalised units)
U_MIN = -1.2 * np.ones(4)
U_MAX =  1.2 * np.ones(4)

# First-order actuator lag  τ_a  [s]
ACTUATOR_TAU = 0.08

# IMU noise standard deviation [rad/s]
SIGMA_OMEGA_DEFAULT = 0.01

# ── Confined-environment scenario definitions ──────────────────────────────────
# Each scenario specifies wall/ground stand-off distances, disturbance scaling
# coefficients, and gain multipliers for the four disturbance components.
SCENARIOS: dict[str, dict] = {
    "nominal": {
        "label": "S1 Nominal corridor",
        "wall_base": 0.24, "wall_amp": 0.04,
        "ground_base": 0.28, "ground_amp": 0.03,
        "wall_coeff": 0.018, "ground_coeff": 0.014,
        "wall_scale": 1.00, "ground_scale": 1.00,
        "residual_scale": 1.00, "jet_scale": 1.00,
    },
    "narrow_corridor": {
        "label": "S2 Narrow corridor",
        "wall_base": 0.18, "wall_amp": 0.03,
        "ground_base": 0.30, "ground_amp": 0.03,
        "wall_coeff": 0.022, "ground_coeff": 0.014,
        "wall_scale": 1.35, "ground_scale": 0.95,
        "residual_scale": 1.05, "jet_scale": 1.05,
    },
    "seabed_hover": {
        "label": "S3 Seabed-hover",
        "wall_base": 0.30, "wall_amp": 0.03,
        "ground_base": 0.18, "ground_amp": 0.02,
        "wall_coeff": 0.016, "ground_coeff": 0.020,
        "wall_scale": 0.95, "ground_scale": 1.40,
        "residual_scale": 1.08, "jet_scale": 0.90,
    },
    "compound_boundary": {
        "label": "S4 Compound boundary",
        "wall_base": 0.19, "wall_amp": 0.03,
        "ground_base": 0.20, "ground_amp": 0.02,
        "wall_coeff": 0.022, "ground_coeff": 0.018,
        "wall_scale": 1.30, "ground_scale": 1.25,
        "residual_scale": 1.18, "jet_scale": 1.35,
    },
}

GENERALIZATION_SCENARIOS = [
    "nominal", "narrow_corridor", "seabed_hover", "compound_boundary"
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def make_g_actual(mismatch_pct: float = 0.10) -> np.ndarray:
    """Return a true effectiveness matrix with the requested gain-mismatch level.

    mismatch_pct = 0.10  →  default G_ACTUAL_DEFAULT (10 % mismatch).
    mismatch_pct = 0.00  →  G_NOM  (perfect knowledge).
    """
    ratio = mismatch_pct / 0.10
    base_scale = np.diag([1.00, 0.92, 1.05, 0.95])
    base_additive = np.array([
        [ 0.03,  0.00, -0.02,  0.01],
        [ 0.00, -0.02,  0.01,  0.02],
        [ 0.02, -0.01,  0.00, -0.02],
    ])
    scale = np.eye(4) + ratio * (base_scale - np.eye(4))
    return G_NOM @ scale + ratio * base_additive


def cross_term(omega: np.ndarray) -> np.ndarray:
    """Coriolis / gyroscopic term  ω × (I ω)."""
    return np.cross(omega, INERTIA @ omega)


def clip_norm(x: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(x, -limit, limit)


def max_available_torque_norm(mismatch_pct: float = 0.10) -> float:
    """Brute-force maximum torque norm over all actuator corner combinations."""
    g = make_g_actual(mismatch_pct)
    corners = np.array(np.meshgrid(*[[U_MIN[0], U_MAX[0]]] * 4)).T.reshape(-1, 4)
    return float(np.max(np.linalg.norm((g @ corners.T).T, axis=1)))


# ── Reference attitude profiles ────────────────────────────────────────────────
def reference_step(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step command [20, -10, 15] deg applied at t = 2 s."""
    eta_ref = np.zeros(3)
    if t >= 2.0:
        eta_ref = np.deg2rad(np.array([20.0, -10.0, 15.0]))
    return eta_ref, np.zeros(3), np.zeros(3)


def reference_hold(_t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Constant-hold reference [10, -8, 12] deg for disturbance-rejection tests."""
    return np.deg2rad(np.array([10.0, -8.0, 12.0])), np.zeros(3), np.zeros(3)


def reference_sinusoidal(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothly varying sinusoidal reference for continuous-tracking tests.

    φ_r = 5 sin(0.5 t) deg
    θ_r = 3 sin(0.8 t + π/4) deg
    ψ_r = 4 sin(0.3 t − π/6) deg
    """
    eta_ref = np.deg2rad(np.array([
        5.0 * np.sin(0.5 * t),
        3.0 * np.sin(0.8 * t + np.pi / 4),
        4.0 * np.sin(0.3 * t - np.pi / 6),
    ]))
    eta_ref_dot = np.deg2rad(np.array([
        5.0 * 0.5  * np.cos(0.5 * t),
        3.0 * 0.8  * np.cos(0.8 * t + np.pi / 4),
        4.0 * 0.3  * np.cos(0.3 * t - np.pi / 6),
    ]))
    eta_ref_ddot = np.deg2rad(np.array([
        -5.0 * 0.25 * np.sin(0.5 * t),
        -3.0 * 0.64 * np.sin(0.8 * t + np.pi / 4),
        -4.0 * 0.09 * np.sin(0.3 * t - np.pi / 6),
    ]))
    return eta_ref, eta_ref_dot, eta_ref_ddot


def get_reference(case: str, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the appropriate reference profile based on the simulation case."""
    if case == "step":
        return reference_step(t)
    elif case in ("disturbance",):
        return reference_hold(t)
    elif case == "combined":
        return reference_step(t)          # step at t=2 s, disturbance at t=4 s
    elif case == "sinusoidal_dist":
        return reference_sinusoidal(t)
    else:
        return reference_hold(t)


# ── Confined-environment disturbance model ─────────────────────────────────────
def disturbance_profile(
    t: float,
    eta: np.ndarray,
    omega: np.ndarray,
    case: str,
    scenario: str = "nominal",
    freq_scale: float = 1.0,
    dist_onset: float = 8.0,
) -> np.ndarray:
    """Compute the total external disturbance torque at time t.

    The model decomposes the disturbance into four additive components:

      τ_ext = τ_wall + τ_ground + τ_residual + τ_jet

    where τ_wall and τ_ground follow an inverse-square proximity scaling,
    τ_residual captures unmodelled hydrodynamics, and τ_jet is a structured
    jet disturbance injected after dist_onset in disturbance/combined cases.

    Parameters
    ----------
    t          : current simulation time [s]
    eta        : Euler-angle state [φ, θ, ψ] [rad]
    omega      : body angular-velocity state [p, q, r] [rad/s]
    case       : simulation case ('step', 'disturbance', 'combined', …)
    scenario   : key into SCENARIOS dict
    freq_scale : multiplicative scaling of all sinusoidal frequencies
    dist_onset : time at which the jet disturbance is injected [s]
    """
    cfg = SCENARIOS[scenario]

    # --- wall proximity effect ---
    d_wall   = cfg["wall_base"]   + cfg["wall_amp"]   * np.sin(0.22 * freq_scale * t)
    d_ground = cfg["ground_base"] + cfg["ground_amp"] * np.cos(0.17 * freq_scale * t)
    k_wall   = cfg["wall_coeff"]   / (d_wall   + 0.08) ** 2
    k_ground = cfg["ground_coeff"] / (d_ground + 0.08) ** 2

    tau_wall = cfg["wall_scale"] * np.array([
        k_wall * ( 0.6 * omega[1] + 0.4 * np.sin(eta[2])),
        k_wall * (-0.5 * omega[0] + 0.2 * np.sin(eta[0])),
        k_wall *   0.25 * np.sin(eta[1]),
    ])

    # --- ground proximity effect ---
    tau_ground = cfg["ground_scale"] * np.array([
        k_ground *  0.15 * np.sin(eta[1]),
        k_ground * ( 0.4 * omega[0] - 0.2 * np.sin(eta[2])),
        k_ground * (-0.35 * omega[1] + 0.1 * np.sin(eta[0])),
    ])

    # --- unmodelled hydrodynamic residual ---
    tau_res = cfg["residual_scale"] * np.array([
        0.030 * np.sin(1.10 * freq_scale * t),
        0.020 * np.cos(0.90 * freq_scale * t + 0.3),
        0.025 * np.sin(0.70 * freq_scale * t + 0.6),
    ])

    # --- structured jet / vortex disturbance ---
    tau_jet = np.zeros(3)
    if case in ("disturbance", "combined", "sinusoidal_dist") and t >= dist_onset:
        dt = t - dist_onset
        tau_jet = cfg["jet_scale"] * np.array([
             0.12 * np.sin(1.00 * freq_scale * dt),
            -0.10 * np.sin(0.80 * freq_scale * dt),
             0.08 * np.cos(0.90 * freq_scale * dt),
        ])

    return tau_wall + tau_ground + tau_res + tau_jet


# ── Inner-loop: regularised fin allocator ─────────────────────────────────────
def allocate_torque(
    tau_ref: np.ndarray,
    u_prev: np.ndarray,
    lambda_reg: float,
    smooth_weight: float,
    allocation_matrix: np.ndarray = G_NOM,
) -> np.ndarray:
    """Regularised weighted least-squares torque-to-fin allocation.

    Solves:
        min_{Δu}  ‖τ_ref − G Δu‖²  +  λ ‖Δu‖²  +  μ ‖Δu − u_prev‖²
    subject to:  U_MIN ≤ u ≤ U_MAX

    The analytical solution (before clipping) is:
        Δu* = (G^T G + (λ+μ) I)^{-1} (G^T τ_ref + μ u_prev)
    """
    gram = allocation_matrix.T @ allocation_matrix + (lambda_reg + smooth_weight) * np.eye(4)
    rhs  = allocation_matrix.T @ tau_ref + smooth_weight * u_prev
    u_cmd = np.linalg.solve(gram, rhs)
    return np.clip(u_cmd, U_MIN, U_MAX)
