from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from controllers.deepc import DeePCConfig, DeePCController
from data import Trajectory


@dataclass
class SelectDPCConfig(DeePCConfig):
    n_closest: int = 10


class SelectDPCController(DeePCController):
    def __init__(self, trajectories: List[Trajectory], config: SelectDPCConfig) -> None:
        self.select_config = config
        super().__init__(trajectories=trajectories, config=config)

    def _select_trajectories(self, x_now: float) -> List[Trajectory]:
        distances = []
        for tr in self.trajectories:
            # Distance to the closest sampled state on each trajectory.
            dist = float(np.min(np.abs(tr.x - x_now)))
            distances.append(dist)

        idx = np.argsort(np.asarray(distances))
        n_pick = int(max(1, min(self.select_config.n_closest, len(self.trajectories))))
        selected = [self.trajectories[i] for i in idx[:n_pick]]
        return selected

    def compute_control(
        self,
        x_now: float,
        u_hist: np.ndarray,
        y_hist_for_input_times: np.ndarray,
        y_now: float,
        r_future: np.ndarray,
    ) -> float:
        selected = self._select_trajectories(x_now)
        try:
            up, uf, yp, yf = self._build_hankels(selected)
        except ValueError:
            up, uf, yp, yf = self.up, self.uf, self.yp, self.yf

        return self._solve_with_hankels(
            up=up,
            uf=uf,
            yp=yp,
            yf=yf,
            u_hist=u_hist,
            y_hist_for_input_times=y_hist_for_input_times,
            y_now=y_now,
            r_future=r_future,
        )
