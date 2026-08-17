"""
Action Agent: Redis-integrated wrapper for Phase 3.
Subscribes to perception.out & cognition.out, publishes action.out.
"""

import json
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict
from scipy.stats import beta
from .physics import SpacecraftConfig, PhysicsSimulator
from .counterfactual import CounterfactualEngine
from scipy.spatial.transform import Rotation as SciRotation

def clopper_pearson_upper_bound(n_successes: int, n_trials: int,
                                 confidence: float = 0.99) -> float:
    """One-sided exact upper confidence bound on a binomial proportion."""
    if n_trials <= 0:
        return 1.0
    if n_successes <= 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / n_trials)
    if n_successes >= n_trials:
        return 1.0
    return float(beta.ppf(confidence, n_successes + 1, n_trials - n_successes))

class ActionAgent:
    def __init__(self, config: SpacecraftConfig,
                 redis_host: str = 'localhost', redis_port: int = 6379,
                 standalone: bool = False, n_mc: int = 100):
        self.cfg = config
        self.standalone = standalone
        self.engine = CounterfactualEngine(config, n_mc=n_mc)
        self.latest_pose: Optional[Dict] = None
        self.latest_situation: Optional[Dict] = None

        if not standalone:
            try:
                import redis as redis_lib
                self.r = redis_lib.Redis(host=redis_host, port=redis_port,
                                         decode_responses=True)
                self.pubsub = self.r.pubsub()
                self.pubsub.subscribe('perception.out', 'cognition.out', 'human.in')
                print(f"[ActionAgent] Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                print(f"[ActionAgent] Redis connection failed ({e}), falling back to standalone.")
                self.standalone = True
                self.r = None
        else:
            self.r = None

    def process_one(self, message_json: str) -> Optional[Dict]:
        """Process a single message (for standalone or threaded loop)."""
        try:
            msg = json.loads(message_json)
        except json.JSONDecodeError:
            return None

        agent_id = msg.get('agent_id')
        mtype = msg.get('message_type')

        if agent_id == 'perception' and mtype == 'pose_estimate':
            self.latest_pose = msg['payload']
            return None  # Wait for situation or explicit trigger

        if agent_id == 'cognition' and mtype == 'situation_vector':
            self.latest_situation = msg['payload']
            return self._generate_and_publish()

        if agent_id == 'human' and mtype == 'override':
            # Armstrong Protocol: log and act (simplified)
            print(f"[ActionAgent] Human override received: {msg['payload']}")
            return None

        return None

    def _generate_and_publish(self) -> Optional[Dict]:
        if self.latest_pose is None:
            print("[ActionAgent] No pose available, skipping.")
            return None

        state0 = self._build_initial_state()
        situation = self.latest_situation or {}
        results = self.engine.evaluate_all_actions(state0, situation)
        payload = self._format_payload(results)

        if self.r:
            self.r.publish('action.out', json.dumps(payload))
            print(f"[ActionAgent] Published recommendation: {payload['payload']['recommended_action']}")

        return payload

    def _build_initial_state(self):
        pose = {
            'translation': self.latest_pose['pose']['t'],
            'quaternion': self._rotmat_to_quat(self.latest_pose['pose']['R']),
            'sigma_t': self.latest_pose['uncertainty'].get('sigma_t', 0.05),
            'hopf_grid': self.latest_pose['uncertainty'].get('hopf_grid')
        }
        habitat = {}
        if self.latest_situation and 'subsystem_states' in self.latest_situation:
            habitat = self.latest_situation['subsystem_states']
        return self.engine.twin.sim.initialize_state(pose, habitat)

    @staticmethod
    def _rotmat_to_quat(R_mat):
        R_arr = np.array(R_mat, dtype=float)
        q_scipy = SciRotation.from_matrix(R_arr).as_quat()  # [x,y,z,w]
        return np.roll(q_scipy, 1).tolist()  # [w,x,y,z]

    def _format_payload(self, results: list) -> Dict:
            actions_out = []
            for r in results:
                p_mean = r['metrics']['tactical']['collision_probability']
                p_std = r['metrics']['tactical']['collision_probability_std']
                n_mc = r['metrics']['tactical']['trajectories'].shape[0]
                n_collisions = int(round(p_mean * n_mc))
                p_upper_99 = clopper_pearson_upper_bound(n_collisions, n_mc, confidence=0.99)

                actions_out.append({
                    'name': r['action'],
                    'score': round(r['score'], 4),
                    'metrics': {
                        'collision_probability': {
                            'mean': round(p_mean, 4),
                            'std': round(p_std, 4),
                            'guaranteed_upper_bound_99pct': round(p_upper_99, 4),
                            'n_mc': n_mc,
                        },
                        'final_soc': {
                            'mean': round(r['metrics']['strategic']['final_soc_mean'], 4),
                            'std': round(r['metrics']['strategic']['final_soc_std'], 4)
                        }
                    }
                })
