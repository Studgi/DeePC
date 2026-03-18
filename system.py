from __future__ import annotations

import numpy as np


class NonlinearSystem:
    def __init__(self, u_min: float = -1.0, u_max: float = 1.0) -> None:
        self.u_min = u_min
        self.u_max = u_max

    def step(self, x: float, u: float) -> float:
        u_clipped = float(np.clip(u, self.u_min, self.u_max))
        return float(x + 0.5 * np.sin(x) + u_clipped)

    def output(self, x: float) -> float:
        return float(x)
