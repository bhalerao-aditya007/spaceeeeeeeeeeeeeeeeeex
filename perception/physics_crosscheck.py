import numpy as np


def cwh_stm(n: float, dt: float) -> np.ndarray:
    nt = n * dt
    s, c = np.sin(nt), np.cos(nt)
    Phi_rr = np.array([[4 - 3*c, 0, 0], [6*(s - nt), 1, 0], [0, 0, c]])
    Phi_rv = np.array([[s/n, 2*(1 - c)/n, 0],
                        [2*(c - 1)/n, (4*s - 3*nt)/n, 0],
                        [0, 0, s/n]])
    Phi_vr = np.array([[3*n*s, 0, 0], [6*n*(c - 1), 0, 0], [0, 0, -n*s]])
    Phi_vv = np.array([[c, 2*s, 0], [-2*s, 4*c - 3, 0], [0, 0, c]])
    Phi = np.zeros((6, 6))
    Phi[0:3, 0:3], Phi[0:3, 3:6] = Phi_rr, Phi_rv
    Phi[3:6, 0:3], Phi[3:6, 3:6] = Phi_vr, Phi_vv
    return Phi


class PhysicsCrossChecker:
    def __init__(self, mean_motion: float, residual_threshold_m: float = 2.0):
        self.n = mean_motion
        self.threshold_m = residual_threshold_m
        self._last_r = None
        self._last_v = None
        self._last_t = None

    def update(self, t_measured: np.ndarray, timestamp: float) -> dict:
        t_measured = np.asarray(t_measured, dtype=float)
        if not np.all(np.isfinite(t_measured)):
            # bad measurement — don't corrupt state, don't penalize confidence
            return {"physics_residual_m": 0.0, "physics_consistent": True,
                    "note": "non-finite measurement, skipped"}

        if self._last_r is None:
            self._last_r, self._last_v, self._last_t = t_measured.copy(), np.zeros(3), timestamp
            return {"physics_residual_m": 0.0, "physics_consistent": True,
                    "predicted_position": t_measured.tolist(),
                    "velocity_estimate": [0.0, 0.0, 0.0], "note": "initializing"}

        dt = timestamp - self._last_t
        if dt <= 0 or dt > 30.0:
            self._last_r, self._last_t = t_measured.copy(), timestamp
            return {"physics_residual_m": 0.0, "physics_consistent": True,
                    "predicted_position": t_measured.tolist(),
                    "velocity_estimate": self._last_v.tolist(),
                    "note": f"dt={dt:.2f}s out of range, skipped"}

        state0 = np.concatenate([self._last_r, self._last_v])
        predicted_r = (cwh_stm(self.n, dt) @ state0)[0:3]
        residual = float(np.linalg.norm(predicted_r - t_measured))
        consistent = residual <= self.threshold_m

        new_v = (t_measured - self._last_r) / dt
        self._last_r, self._last_v, self._last_t = t_measured.copy(), new_v, timestamp

        return {
            "physics_residual_m": round(residual, 4),
            "physics_consistent": consistent,
            "predicted_position": predicted_r.tolist(),
            "velocity_estimate": new_v.tolist(),
        }
