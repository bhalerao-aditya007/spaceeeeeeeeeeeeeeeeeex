"""
Full system integration — connects all 5 agents.
Handles message format translation between phases.
"""

import sys
import os
import json
import time
import threading
import numpy as np
from action.agent import clopper_pearson_upper_bound
import redis
from orchestrator.redis_fallback import get_redis_client

# ── Perception ──────────────────────────────────────────────
from perception.perception_agent import PerceptionAgent

# ── Cognition (Phase 2) ──────────────────────────────────────
from cognition.cognition_agent import (
    HyperdimensionalCognitionLayer,
    PoseEstimate as HDCPoseEstimate,
    Telemetry, AnomalyReport, DomainContext
)

# ── Action (Phase 3) ─────────────────────────────────────────
from action.physics import default_spacecraft_config
from action.counterfactual import CounterfactualEngine

# ── Orchestrator ─────────────────────────────────────────────
from orchestrator.orchestrator import Orchestrator
from orchestrator.message_schemas import (
    PoseEstimateMessage, SituationVectorMessage,
    ActionRecommendationMessage, HumanOverrideMessage,
    OverrideLevel, ActionType
)

REDIS_HOST = "localhost"
REDIS_PORT = 6379
MODEL_PATH = "perception/checkpoints/best.pt"


# ══════════════════════════════════════════════════════════════
# MESSAGE ADAPTERS — translate between phase formats
# ══════════════════════════════════════════════════════════════

def perception_output_to_redis_msg(output) -> PoseEstimateMessage:
    """Convert PerceptionAgent output to orchestrator PoseEstimateMessage."""
    return PoseEstimateMessage(
        R=output.pose.R,
        t=output.pose.t,
        quaternion=output.pose.quaternion,
        jensen_gain=output.uncertainty.jensen_gain,
        confidence_level=output.uncertainty.confidence_level,
        confidence_label=output.uncertainty.confidence_label,
        sigma_R_deg=output.uncertainty.sigma_R_deg,
        sigma_t_m=output.uncertainty.sigma_t_m,
        nearest_anchor_idx=output.uncertainty.nearest_anchor_idx,
        anchor_distance_deg=output.uncertainty.anchor_distance_deg,
        is_trustworthy=output.is_trustworthy,
        physics_residual_m=output.uncertainty.physics_residual_m,      
        physics_consistent=output.uncertainty.physics_consistent,
        processing_time_ms=output.metadata["processing_time_ms"],
        image_shape=output.metadata["image_shape"]
    )


def perception_msg_to_hdc_input(msg: PoseEstimateMessage,
                                 telemetry_data: dict = None) -> tuple:
    """Convert perception message to Phase 2 HDC inputs."""
    pose = HDCPoseEstimate(
        translation=np.array(msg.t),
        rotation=np.array(msg.R),
        confidence=msg.confidence_level,
        jensen_gain=msg.jensen_gain,
        sigma_t=msg.sigma_t_m,
        sigma_R=msg.sigma_R_deg
    )

    if not telemetry_data:
        print("[WARN] No telemetry data provided to cognition adapter — "
              "using default nominal values")
    tel = Telemetry(
        o2_level=telemetry_data.get("o2_level", 95.0) if telemetry_data else 95.0,
        co2_level=telemetry_data.get("co2_level", 0.5) if telemetry_data else 0.5,
        pressure_kpa=telemetry_data.get("pressure_kpa", 101.3) if telemetry_data else 101.3,
        battery_pct=telemetry_data.get("battery_pct", 87.0) if telemetry_data else 87.0,
        radiator_efficiency_pct=telemetry_data.get("radiator_efficiency_pct", 100.0) if telemetry_data else 100.0,
        coolant_flow_lpm=telemetry_data.get("coolant_flow_lpm", 10.0) if telemetry_data else 10.0,
    )

    anomaly_type = telemetry_data.get("anomaly_type", "none") if telemetry_data else "none"
    severity = telemetry_data.get("severity", "nominal") if telemetry_data else "nominal"
    anomaly = AnomalyReport(
        failure_type=anomaly_type,
        severity=severity,
        propagation_risk="high" if severity == "critical" else "low"
    )

    return pose, tel, anomaly


def hdc_output_to_redis_msg(hdc_result: dict) -> SituationVectorMessage:
    """Convert Phase 2 HDC output to orchestrator SituationVectorMessage."""
    payload = hdc_result.get("payload", {})
    anomaly_meta = payload.get("anomaly_meta", {})

    action_map = {
        "HOLD_POSITION": "hold_position",
        "PROBE_MINIMAL_RISK": "proceed_slow",
        "ABORT": "abort",
        "PROCEED_SLOW": "proceed_slow",
        "PROCEED_NORMAL": "proceed_normal",
    }
    raw_action = payload.get("recommended_action", "HOLD_POSITION")
    mapped_action = action_map.get(raw_action, "hold_position")

    failure_type = anomaly_meta.get("failure_type", "none")
    anomaly_detected = failure_type not in ("none", "nominal", "")
    severity = anomaly_meta.get("severity", "nominal")
    narrative = payload.get("explanation", {}).get("narrative", "")

    return SituationVectorMessage(
        situation_id=f"sit_{int(time.time())}",
        anomaly_detected=anomaly_detected,
        anomaly_type=failure_type,
        anomaly_severity=severity,
        novelty_score=float(1.0 - payload.get("max_similarity", 0.5)),
        similar_case_id=str(payload.get("nearest_cases", [{}])[0].get("case_id", "")),
        similar_case_outcome=payload.get("nearest_cases", [{}])[0].get("outcome", ""),
        recommended_action=mapped_action,
        action_confidence=float(payload.get("max_similarity", 0.5)),
        explanation=narrative[:200] if narrative else "HDC nominal"
    )


def action_result_to_redis_msg(results: list) -> ActionRecommendationMessage:
    """Convert Phase 3 counterfactual results to orchestrator message."""
    if not results:
        return ActionRecommendationMessage(
            primary_action="hold_position",
            primary_score=0.5,
            collision_prob=0.0,
            mission_success_prob=0.5,
            resource_cost=0.2,
            alternatives=[],
            simulation_horizon_s=60,
            mc_runs=0,
            explanation="No results"
        )

    action_map = {
        "ABORT": "abort",
        "HOLD": "hold_position",
        "PROCEED_SLOW": "proceed_slow",
        "PROCEED_NORMAL": "proceed_normal",
        "RECONFIGURE_POWER": "reconfigure_power",
        "ISOLATE_MODULE": "isolate_module",
        "EMERGENCY_VENT": "emergency_vent",
    }

    best = results[0]
    collision = best["metrics"]["tactical"]["collision_probability"]
    n_mc_best = best["metrics"]["tactical"]["trajectories"].shape[0]
    n_coll_best = int(round(collision * n_mc_best))
    collision_upper = clopper_pearson_upper_bound(n_coll_best, n_mc_best, 0.99)
    mapped = action_map.get(best["action"], "hold_position")

    alternatives = []
    for r in results[1:4]:
        alternatives.append({
            "action": action_map.get(r["action"], r["action"]),
            "score": round(r["score"], 3),
            "collision_prob": round(r["metrics"]["tactical"]["collision_probability"], 3)
        })

    return ActionRecommendationMessage(
        primary_action=mapped,
        primary_score=round(best["score"], 3),
        collision_prob=round(collision, 3),
        collision_prob_upper_bound_99=round(collision_upper, 3),
        mission_success_prob=round(1.0 - collision, 3),
        resource_cost=0.2,
        alternatives=alternatives,
        simulation_horizon_s=60,
        mc_runs=100,
        explanation=f"Monte Carlo 100 runs, best={best['action']}, "
                    f"collision={collision:.3f}"
    )


# ══════════════════════════════════════════════════════════════
# AGENT RUNNERS
# ══════════════════════════════════════════════════════════════

class IntegratedSystem:
    """Runs all 5 agents in one process with proper message passing."""

    def __init__(self):
        self.r = get_redis_client(host=REDIS_HOST, port=REDIS_PORT, db=0)
        self._running = False
        self.telemetry_state = {}

        print("Initializing all agents...")

        # Perception
        print("  [1/4] Loading perception agent...")
        self.perception = PerceptionAgent(
            model_path=MODEL_PATH,
            run_jensen_gain=True,
            n_elevation=64,
            n_inplane=16,
            n_jensen_rotations=8
        )

        # Cognition (Phase 2)
        print("  [2/4] Initializing HDC cognition layer...")
        self.hdc = HyperdimensionalCognitionLayer(config={
            "dim": 10000,
            "similarity_threshold": 0.55,
            "novelty_threshold": 0.45
        })
        self._seed_hdc_memory()

        # Action (Phase 3)
        print("  [3/4] Initializing digital twin...")
        self.spacecraft_cfg = default_spacecraft_config()
        self.counterfactual = CounterfactualEngine(
            self.spacecraft_cfg, n_mc=20
        )

        # Orchestrator
        print("  [4/4] Starting orchestrator...")
        self.orchestrator = Orchestrator()

        print("All agents ready.")

    def _seed_hdc_memory(self):
        """Seed associative memory with known cases."""
        known_cases = [
            {
                "pose": HDCPoseEstimate(
                    translation=np.array([10.0, 0.0, 0.0]),
                    rotation=np.eye(3),
                    confidence="high",
                    jensen_gain=1.5
                ),
                "tel": Telemetry(o2_level=95.0, battery_pct=87.0,
                                 radiator_efficiency_pct=100.0),
                "anomaly": AnomalyReport("none", "nominal", "low"),
                "action": "HOLD_POSITION",
                "outcome": "success",
                "success_rate": 95.0
            },
            {
                "pose": HDCPoseEstimate(
                    translation=np.array([12.0, 0.0, 0.0]),
                    rotation=np.eye(3),
                    confidence="low",
                    jensen_gain=2.8
                ),
                "tel": Telemetry(o2_level=94.0, battery_pct=85.0,
                                 radiator_efficiency_pct=45.0),
                "anomaly": AnomalyReport("thermal_failure", "critical", "high"),
                "action": "HOLD_POSITION",
                "outcome": "success",
                "success_rate": 91.0
            },
            {
                "pose": HDCPoseEstimate(
                    translation=np.array([5.0, 0.0, 0.0]),
                    rotation=np.eye(3),
                    confidence="moderate",
                    jensen_gain=1.0
                ),
                "tel": Telemetry(o2_level=90.0, battery_pct=40.0,
                                 radiator_efficiency_pct=80.0),
                "anomaly": AnomalyReport("power_loss", "degraded", "medium"),
                "action": "RECONFIGURE_POWER",
                "outcome": "success",
                "success_rate": 88.0
            },
        ]

        for case in known_cases:
            result = self.hdc.process(
                pose_estimate=case["pose"],
                telemetry=case["tel"],
                anomaly_report=case["anomaly"],
                mission_phase="approach"
            )
            self.hdc.learn_outcome(
                situation_vector_b64=result["payload"]["situation_vector_b64"],
                action_taken=case["action"],
                outcome=case["outcome"],
                success_rate=case["success_rate"]
            )

        print(f"  HDC memory seeded with {len(known_cases)} historical cases")

    def run_full_pipeline(self, image: np.ndarray,
                          telemetry: dict = None,
                          label: str = "") -> dict:
        """Run complete pipeline for one frame. Returns all agent outputs."""
        results = {}
        telemetry = telemetry or {}

        # ── Step 1: Perception ──────────────────────────────────────────
        perc_output = self.perception.predict(image)
        perc_msg = perception_output_to_redis_msg(perc_output)
        self.r.publish("perception.out", perc_msg.to_json())

        results["perception"] = {
            "jensen_gain": perc_output.uncertainty.jensen_gain,
            "confidence": perc_output.uncertainty.confidence_level,
            "trustworthy": perc_output.is_trustworthy,
            "translation": perc_output.pose.t
        }

        # ── Step 2: Cognition (HDC) ─────────────────────────────────────
        pose_hdc, tel, anomaly = perception_msg_to_hdc_input(perc_msg, telemetry)
        hdc_result = self.hdc.process(
            pose_estimate=pose_hdc,
            telemetry=tel,
            anomaly_report=anomaly,
            mission_phase=telemetry.get("mission_phase", "approach"),
            domain=DomainContext(
                lighting=telemetry.get("lighting", "nominal"),
                background="deep_space"
            )
        )
        sit_msg = hdc_output_to_redis_msg(hdc_result)
        self.r.publish("cognition.out", sit_msg.to_json())

        results["cognition"] = {
            "is_novel": hdc_result["payload"]["is_novel"],
            "max_similarity": hdc_result["payload"]["max_similarity"],
            "strategy": hdc_result["payload"]["strategy"],
            "recommended": hdc_result["payload"]["recommended_action"],
            "narrative": hdc_result["payload"]["explanation"]["narrative"][:100]
        }

        # ── Step 3: Action (Digital Twin) ───────────────────────────────
        t_vec = np.array(perc_msg.t)
        q_vec = np.array(perc_msg.quaternion)

        # Clamp translation to realistic orbital range — LOG the fallback
        t_norm = np.linalg.norm(t_vec)
        _translation_was_clamped = False
        if t_norm < 1e-3 or t_norm > 1000.0:
            print(f"[WARN] Translation out of range (norm={t_norm:.4f}m), "
                  f"falling back to safe default [10, 0, 0]")
            t_vec = np.array([10.0, 0.0, 0.0])
            _translation_was_clamped = True

        pose_for_twin = {
            "translation": t_vec,
            "quaternion": q_vec,
            "sigma_t": perc_msg.sigma_t_m,
        }
        state0 = self.counterfactual.twin.sim.initialize_state(pose_for_twin, {})

        situation_for_twin = {
            "novelty_score": hdc_result["payload"]["max_similarity"]
        }
        action_results = self.counterfactual.evaluate_all_actions(
            state0, situation_for_twin
        )
        action_msg = action_result_to_redis_msg(action_results)
        self.r.publish("action.out", action_msg.to_json())

        results["action"] = {
            "recommended": action_msg.primary_action,
            "score": action_msg.primary_score,
            "collision_prob": action_msg.collision_prob,
            "explanation": action_msg.explanation,
            "translation_fallback_used": _translation_was_clamped
        }

        results["label"] = label
        return results

    def start(self):
        self.orchestrator.start()
        self._running = True
        time.sleep(0.5)

    def stop(self):
        self.orchestrator.stop()
        self._running = False


# ══════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════

def load_image(path: str) -> np.ndarray:
    """
    Load a real test image from disk.
    Requires actual spacecraft images in perception/test_images/.
    """
    from PIL import Image
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Test image not found: {path}. "
            f"Place real spacecraft images in perception/test_images/"
        )
    img = np.array(Image.open(path).convert("RGB").resize((224, 224)))
    return img


def print_results(results: dict):
    p = results["perception"]
    c = results["cognition"]
    a = results["action"]

    trust = "[TRUSTED]" if p["trustworthy"] else "[UNTRUSTED]"
    print(f"\n  PERCEPTION  {trust} JG={p['jensen_gain']:.1f}deg conf={p['confidence']}")
    print(f"  COGNITION   novel={c['is_novel']} sim={c['max_similarity']:.2f} -> {c['recommended']}")
    print(f"  ACTION      {a['recommended']} score={a['score']:.2f} collision={a['collision_prob']:.3f}")
    print(f"  NARRATIVE   {c['narrative']}...")


if __name__ == "__main__":
    print("=" * 60)
    print("FULL SYSTEM INTEGRATION -- ALL 5 AGENTS")
    print("=" * 60)

    # Connect Redis (or in-memory fallback)
    r = get_redis_client(host=REDIS_HOST, port=REDIS_PORT, db=0)
    print("[OK] PubSub Messaging ready")

    system = IntegratedSystem()
    system.start()

    scenarios = [
        ("nominal", {},                                   "Nominal approach"),
        ("glare",   {"lighting": "glare"},                "Sun glare — pose uncertain"),
        ("nominal", {"anomaly_type": "thermal_failure",
                     "severity": "critical",
                     "radiator_efficiency_pct": 30.0},   "Thermal anomaly"),
        ("dark",    {"anomaly_type": "power_loss",
                     "severity": "degraded",
                     "battery_pct": 35.0},               "Power degraded + dark"),
    ]

    try:
        for img_type, tel, label in scenarios:
            print(f"\n{'='*60}")
            print(f"SCENARIO: {label}")
            print(f"{'='*60}")
            img = load_image(os.path.join("perception", "test_images", f"{img_type}.png"))
            results = system.run_full_pipeline(img, tel, label)
            print_results(results)
            time.sleep(2.0)

    finally:
        system.stop()
        print("\n" + "=" * 60)
        print("INTEGRATION COMPLETE")
        print("=" * 60)
