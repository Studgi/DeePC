from __future__ import annotations

import numpy as np


class KinematicBicycleYaw:
    def __init__(self, u_min: float = -np.pi/4, u_max: float = np.pi/4, v: float = 2.0, L: float = 2.5, dt: float = 0.1) -> None:
        self.u_min = u_min
        self.u_max = u_max
        self.v = v
        self.L = L
        self.dt = dt

    def step(self, x: float, u: float, v: float = None) -> float:
        # x is yaw angle (theta), u is steering angle (delta)
        velocity = v if v is not None else self.v
        u_clipped = float(np.clip(u, self.u_min, self.u_max))
        # Yaw dynamics: dot{theta} = (v/L) * tan(delta)
        return float(x + self.dt * (velocity / self.L) * np.tan(u_clipped))

    def output(self, x: float) -> float:
        return float(x)


class LiftedKinematicBicycleYaw(KinematicBicycleYaw):
    def __init__(self, u_min: float = -np.pi/4, u_max: float = np.pi/4, v: float = 2.0, L: float = 2.5, dt: float = 0.1) -> None:
        super().__init__(u_min, u_max, v, L, dt)
        # the model simulates the true system, but it receives lifted input!
        # wait, the true system takes delta, the lifted system takes tan(delta).
        
    def step(self, x: float, u_lifted: float, v: float = None) -> float:
        velocity = v if v is not None else self.v
        # u_lifted is tan(delta). We assume it is bounded accordingly.
        tan_u_min = np.tan(self.u_min)
        tan_u_max = np.tan(self.u_max)
        u_lifted_clipped = float(np.clip(u_lifted, tan_u_min, tan_u_max))
        return float(x + self.dt * (velocity / self.L) * u_lifted_clipped)


