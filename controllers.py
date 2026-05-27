"""controllers.py — Outer-loop attitude controllers.

Implements five controllers compared in the paper:
  - proposed   : SMC + integral sliding surface + first-order disturbance observer
  - adrc        : Active disturbance rejection control (second-order ESO)
  - smc         : Conventional sliding mode control
  - lqr         : Linear quadratic regulator (state-feedback)
  - pid         : PID controller

Ablation variants:
  - proposed_no_observer : proposed without the disturbance observer (d_hat ≡ 0)
  - proposed_no_smooth   : proposed without the smoothness regulariser (μ = 0)

All controllers output a desired body torque τ_ref ∈ R³.  The ADRC outer loop
is handled inside the simulation engine because it requires the ESO state that
evolves between time steps; this file only provides the non-ADRC controllers.
"""

from __future__ import annotations

import numpy as np

from auv_model import INERTIA, cross_term, clip_norm

# ── ADRC bandwidth parameters ──────────────────────────────────────────────────
ADRC_OMEGA_O = 12.0   # ESO bandwidth        [rad/s]
ADRC_OMEGA_C =  4.0   # Controller bandwidth [rad/s]


def init_adrc_state() -> np.ndarray:
    """Return zero-initialised ADRC extended-state-observer state vector.

    Layout:  [z1_roll, z2_roll, z1_pitch, z2_pitch, z1_yaw, z2_yaw]
    where z1_i ≈ ω_i  and  z2_i ≈ total disturbance acceleration on axis i.
    """
    return np.zeros(6)


def controller_tau_ref(
    controller: str,
    eta: np.ndarray,
    omega: np.ndarray,
    integ: np.ndarray,
    d_hat: np.ndarray,
    eta_ref: np.ndarray,
    eta_ref_dot: np.ndarray,
    eta_ref_ddot: np.ndarray,
    gain_scale: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the desired torque τ_ref and the sliding/error surface.

    Parameters
    ----------
    controller   : controller key string
    eta          : current Euler angles [φ, θ, ψ] [rad]
    omega        : current body angular velocity [p, q, r] [rad/s]
    integ        : integral of attitude error  ∫ e dt  [rad·s]
    d_hat        : disturbance observer estimate (proposed variants)
    eta_ref      : reference attitude [rad]
    eta_ref_dot  : reference attitude rate [rad/s]
    eta_ref_ddot : reference attitude acceleration [rad/s²]
    gain_scale   : optional dict of multiplicative gain scaling factors
                   {'lambda_scale': float, 'ks_scale': float}

    Returns
    -------
    tau_ref : desired torque vector [N·m]
    surface : sliding surface / tracking error (for recording)
    """
    gs = gain_scale or {}
    error     = eta     - eta_ref
    error_dot = omega   - eta_ref_dot

    # ── Proposed: integral sliding-mode + disturbance observer ────────────────
    if controller in {"proposed", "proposed_no_observer", "proposed_no_smooth"}:
        lam_scale = gs.get("lambda_scale", 1.0)
        ks_scale  = gs.get("ks_scale",     1.0)

        # Gains
        K1  = np.diag([5.0,  4.6,  4.6 ]) * lam_scale   # bandwidth
        K2  = np.diag([0.02, 0.02, 0.02])                # integral action
        Ks  = np.diag([1.25, 1.15, 1.15]) * ks_scale     # switching gain
        phi = np.array([0.08, 0.08, 0.08])               # boundary layer

        surface     = error_dot + K1 @ error + K2 @ integ
        sat_surface = np.clip(surface / phi, -1.0, 1.0)

        # Disturbance feed-forward (zeroed for no-observer ablation)
        obs_term = d_hat if controller in {"proposed", "proposed_no_smooth"} else np.zeros(3)

        omega_dot_ref = (eta_ref_ddot
                         - K1 @ error_dot
                         - K2 @ error
                         - Ks @ sat_surface
                         - obs_term)
        tau_ref = INERTIA @ omega_dot_ref + cross_term(omega)
        return tau_ref, surface

    # ── Conventional SMC ──────────────────────────────────────────────────────
    if controller == "smc":
        K1  = np.diag([2.1, 2.0, 2.0])
        Ks  = np.diag([0.72, 0.68, 0.68])
        phi = np.array([0.15, 0.15, 0.15])

        surface     = error_dot + K1 @ error
        sat_surface = np.clip(surface / phi, -1.0, 1.0)
        omega_dot_ref = eta_ref_ddot - K1 @ error_dot - Ks @ sat_surface
        tau_ref = INERTIA @ omega_dot_ref + cross_term(omega)
        return tau_ref, surface

    # ── PID ───────────────────────────────────────────────────────────────────
    if controller == "pid":
        Kp = np.diag([5.0, 4.6, 4.6])
        Ki = np.diag([0.70, 0.55, 0.55])
        Kd = np.diag([2.2,  2.0,  2.0])
        tau_ref = -(Kp @ error + Ki @ integ + Kd @ error_dot)
        return tau_ref, error

    # ── LQR ───────────────────────────────────────────────────────────────────
    if controller == "lqr":
        K_eta   = np.diag([5.5, 5.0, 5.0])
        K_omega = np.diag([3.0, 2.6, 2.6])
        tau_ref = -(K_eta @ error + K_omega @ omega)
        return tau_ref, error

    # ── ADRC: outer loop is handled in simulation.py; return placeholder ──────
    if controller == "adrc":
        return np.zeros(3), error

    raise ValueError(f"Unknown controller: '{controller}'")


def adrc_outer_loop(
    eta: np.ndarray,
    eta_ref: np.ndarray,
    eta_ref_dot: np.ndarray,
    adrc_z: np.ndarray,
) -> np.ndarray:
    """Compute the ADRC PD control law using the current ESO state.

    τ_ref_i = I_ii * (u0_i − z2_i)  where  u0_i = −kp e_η_i − kd e_ω_i

    Parameters
    ----------
    eta       : current Euler angles [rad]
    eta_ref   : reference angles [rad]
    eta_ref_dot : reference rate [rad/s]
    adrc_z    : ESO state vector [z1_roll, z2_roll, …]  (length 6)
    """
    kp = ADRC_OMEGA_C ** 2
    kd = 2.0 * ADRC_OMEGA_C
    tau_ref = np.zeros(3)
    for i in range(3):
        z1 = adrc_z[2 * i]      # estimated ω_i
        z2 = adrc_z[2 * i + 1]  # estimated disturbance acceleration
        e_eta   = eta[i]   - eta_ref[i]
        e_omega = z1        - eta_ref_dot[i]
        u0 = -kp * e_eta - kd * e_omega
        tau_ref[i] = INERTIA[i, i] * (u0 - z2)
    return tau_ref


def adrc_eso_update(
    adrc_z: np.ndarray,
    omega_meas: np.ndarray,
    tau_nom: np.ndarray,
    dt: float,
) -> np.ndarray:
    """One-step Euler update of the second-order ADRC extended-state observer.

    The ESO tracks:
        ż1_i = z2_i + b0_i τ_nom_i − l1 (z1_i − ω_meas_i)
        ż2_i =               −l2   (z1_i − ω_meas_i)

    with bandwidth-parametric gains l1 = 2 ω_o,  l2 = ω_o².

    Parameters
    ----------
    adrc_z     : current ESO state (length 6), modified in-place
    omega_meas : noisy angular-velocity measurement [rad/s]
    tau_nom    : nominal torque G_NOM @ u [N·m]
    dt         : integration step [s]

    Returns
    -------
    Updated adrc_z (same array, also modified in-place).
    """
    from auv_model import INERTIA_INV  # avoid circular import at module level
    l1 = 2.0 * ADRC_OMEGA_O
    l2 = ADRC_OMEGA_O ** 2
    b0 = np.array([INERTIA_INV[0, 0], INERTIA_INV[1, 1], INERTIA_INV[2, 2]])

    for i in range(3):
        z1   = adrc_z[2 * i]
        z2   = adrc_z[2 * i + 1]
        eps1 = z1 - omega_meas[i]
        adrc_z[2 * i]     += dt * (z2 + b0[i] * tau_nom[i] - l1 * eps1)
        adrc_z[2 * i + 1] += dt * (-l2 * eps1)
    return adrc_z
