from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np


@dataclass
class Trajectory:
    x: np.ndarray  # Shape: (T+1,)
    y: np.ndarray  # Shape: (T+1,)
    u: np.ndarray  # Shape: (T,)


def rollout(system: Any, x0: float, u_seq: np.ndarray) -> Trajectory:
    t_data = int(u_seq.shape[0])
    x = np.zeros(t_data + 1, dtype=float)
    y = np.zeros(t_data + 1, dtype=float)

    x[0] = x0
    y[0] = system.output(x0)

    for t in range(t_data):
        x[t + 1] = system.step(x[t], float(u_seq[t]))
        y[t + 1] = system.output(x[t + 1])

    return Trajectory(x=x, y=y, u=u_seq.astype(float))


def generate_dataset(
    system: Any,
    n_data: int,
    t_data: int,
    u_min: float,
    u_max: float,
    x0_min: float = -2.0,
    x0_max: float = 2.0,
    seed: int = 0,
    lift_input: bool = False,
) -> List[Trajectory]:
    rng = np.random.default_rng(seed)
    trajectories: List[Trajectory] = []

    for _ in range(n_data):
        u_seq = rng.uniform(u_min, u_max, size=t_data)
        x0 = float(rng.uniform(x0_min, x0_max))
        # If lifting, the system actually needs to step with the real input but save tan(u)
        # However, for DPC, we can just say "DPC sees the lifted u" 
        # so u_seq generated here should be considered the *unlifted* delta for reality, 
        # but the trajectory stores the lifted input so DeePC builds Hankel using tan(delta).
        
        # Actually rollout uses system.step(). If we use the original bicycle model, 
        # it expects delta. We pass delta, then lift it.
        tr = rollout(system=system, x0=x0, u_seq=u_seq)
        if lift_input:
            tr.u = np.tan(tr.u)
        trajectories.append(tr)

    return trajectories


def summarize_dataset_ranges(trajectories: List[Trajectory]) -> dict[str, float]:
    if not trajectories:
        raise ValueError("Cannot summarize an empty dataset.")

    u_all = np.concatenate([tr.u for tr in trajectories])
    # Use y_1..y_T for DeePC-aligned output range reporting.
    y_all = np.concatenate([tr.y[1:] for tr in trajectories])

    return {
        "u_min": float(np.min(u_all)),
        "u_max": float(np.max(u_all)),
        "y_min": float(np.min(y_all)),
        "y_max": float(np.max(y_all)),
    }


def build_hankel(signal: np.ndarray, window: int) -> np.ndarray:
    n = signal.shape[0]
    if window > n:
        raise ValueError(f"Window {window} is larger than signal length {n}.")
    cols = n - window + 1
    return np.vstack([signal[i : i + cols] for i in range(window)])


def build_hankel_from_trajectories(
    trajectories: List[Trajectory],
    signal_name: str,
    window: int,
) -> np.ndarray:
    blocks = []
    for tr in trajectories:
        sig = getattr(tr, signal_name)
        if signal_name == "y":
            # Align outputs with inputs as y_{k+1} paired with u_k.
            # For a trajectory y_0..y_T and u_0..u_{T-1}, use y_1..y_T.
            sig = sig[1:]
        if sig.shape[0] >= window:
            blocks.append(build_hankel(sig, window))

    if not blocks:
        raise ValueError("No trajectory is long enough for requested Hankel window.")

    return np.concatenate(blocks, axis=1)


def make_reference(t_sim: int, frequency: float = 0.1) -> np.ndarray:
    t = np.arange(t_sim, dtype=float)
    return np.sin(frequency * t)
