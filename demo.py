"""
End-to-end demo script.
Loads real model, starts orchestrator, feeds images, shows decisions.
"""

import sys
import os
import time
import json
import threading
import numpy as np
import redis

from orchestrator.orchestrator import Orchestrator
from orchestrator.message_schemas import (
    PoseEstimateMessage, HumanOverrideMessage,
    OverrideLevel, ActionType
)
from perception.perception_agent import PerceptionAgent


REDIS_HOST = "localhost"
REDIS_PORT = 6379
MODEL_PATH = "perception/checkpoints/best.pt"


def load_test_image(path: str) -> np.ndarray:
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


class PerceptionPublisher:
    """
    Runs perception agent and publishes to Redis perception.out
    """

    def __init__(self, agent: PerceptionAgent):
        self.agent  = agent
        self.redis  = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        self._running = False

    def publish_once(self, image: np.ndarray, label: str = ""):
        """Run inference on one image and publish result."""
        output = self.agent.predict(image)

        msg = PoseEstimateMessage(
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
            processing_time_ms=output.metadata["processing_time_ms"],
            image_shape=output.metadata["image_shape"]
        )

        self.redis.publish("perception.out", msg.to_json())

        status = "✅ TRUSTED" if output.is_trustworthy else "❌ UNTRUSTED"
        print(f"  [{label}] JG={output.uncertainty.jensen_gain:.1f}° "
              f"conf={output.uncertainty.confidence_level} {status} "
              f"t={output.metadata['processing_time_ms']:.0f}ms")

        return output


def listen_for_decisions(duration: float = 5.0) -> list:
    """Listen to orchestrator consensus channel and collect decisions."""
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    p = r.pubsub()
    p.subscribe("orchestrator.consensus", "orchestrator.escalation")

    decisions = []
    escalations = []
    t_end = time.time() + duration

    for msg in p.listen():
        if time.time() > t_end:
            break
        if msg["type"] != "message":
            continue
        channel = msg["channel"].decode()
        data = json.loads(msg["data"])
        if channel == "orchestrator.consensus":
            decisions.append(data)
        elif channel == "orchestrator.escalation":
            escalations.append(data)

    return decisions, escalations


def run_demo_scenario(name: str,
                      image_scenario: str,
                      agent: PerceptionAgent,
                      publisher: PerceptionPublisher,
                      orc: Orchestrator,
                      n_frames: int = 3):
    """Run one demo scenario."""
    print(f"\n{'='*60}")
    print(f"SCENARIO: {name}")
    print(f"{'='*60}")

    decisions = []
    escalations = []

    def collect():
        d, e = listen_for_decisions(duration=n_frames * 4.0)
        decisions.extend(d)
        escalations.extend(e)

    collector = threading.Thread(target=collect, daemon=True)
    collector.start()

    time.sleep(0.3)

    for i in range(n_frames):
        img = load_test_image(os.path.join("perception", "test_images", f"{image_scenario}.png"))
        publisher.publish_once(img, label=f"frame_{i+1}")
        time.sleep(1.0)

    collector.join(timeout=n_frames * 4.0 + 2)

    print(f"\nRESULTS:")
    print(f"  Decisions received: {len(decisions)}")
    print(f"  Escalations: {len(escalations)}")

    if decisions:
        latest = decisions[-1]
        print(f"  Final action: {latest['final_action']}")
        print(f"  Consensus: {latest['consensus_reached']}")
        print(f"  Reasoning: {latest['reasoning'][:100]}...")

    return decisions, escalations


def run_armstrong_demo(publisher: PerceptionPublisher,
                       orc: Orchestrator):
    """Demo Armstrong Protocol override."""
    print(f"\n{'='*60}")
    print("ARMSTRONG PROTOCOL DEMO — Human Override")
    print(f"{'='*60}")

    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

    decisions = []
    def collect():
        d, _ = listen_for_decisions(duration=5.0)
        decisions.extend(d)

    collector = threading.Thread(target=collect, daemon=True)
    collector.start()
    time.sleep(0.3)

    # Publish a glare image (low confidence)
    img = load_test_image(os.path.join("perception", "test_images", "glare.png"))
    print("Publishing LOW CONFIDENCE frame...")
    publisher.publish_once(img, label="glare_frame")
    time.sleep(1.0)

    # Human sends override
    print("Human sends Level 3 REPLACE override -> PROCEED_SLOW")
    override = HumanOverrideMessage(
        override_level=OverrideLevel.REPLACE,
        selected_action=ActionType.PROCEED_SLOW,
        rationale="Visual confirmation from window — target stable",
        operator_id="commander"
    )
    r.publish("human.in", override.to_json())
    time.sleep(1.0)

    collector.join(timeout=6.0)

    if decisions:
        for d in decisions:
            if d.get("override_applied"):
                print(f"\n✅ Override applied: {d['final_action']}")
                print(f"   Reasoning: {d['reasoning']}")
                break


def main():
    print("SPACECRAFT AUTONOMY SYSTEM — END TO END DEMO")
    print("=" * 60)

    # Check Redis
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        r.ping()
        print("✅ Redis connected")
    except Exception as e:
        print(f"❌ Redis not running: {e}")
        return

    # Load perception agent
    print("\nLoading perception model...")
    try:
        agent = PerceptionAgent(
            model_path=MODEL_PATH,
            run_jensen_gain=True,
            n_elevation=64,
            n_inplane=16,
            n_jensen_rotations=8  # 8 for speed in demo
        )
        print("✅ Perception agent ready")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return

    publisher = PerceptionPublisher(agent)

    # Start orchestrator
    print("\nStarting orchestrator...")
    orc = Orchestrator()
    orc.start()
    time.sleep(0.5)
    print("✅ Orchestrator running")

    # Run scenarios
    try:
        # Scenario 1: Nominal
        run_demo_scenario(
            name="Nominal Approach (high confidence)",
            image_scenario="nominal",
            agent=agent,
            publisher=publisher,
            orc=orc,
            n_frames=3
        )

        # Scenario 2: Sun Glare
        run_demo_scenario(
            name="Sun Glare (pose uncertainty spike)",
            image_scenario="glare",
            agent=agent,
            publisher=publisher,
            orc=orc,
            n_frames=3
        )

        # Scenario 3: Deep Shadow
        run_demo_scenario(
            name="Deep Shadow (sensor degraded)",
            image_scenario="dark",
            agent=agent,
            publisher=publisher,
            orc=orc,
            n_frames=3
        )

        # Scenario 4: Armstrong Override
        run_armstrong_demo(publisher, orc)

    finally:
        orc.stop()
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()