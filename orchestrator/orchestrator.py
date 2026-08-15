"""
Orchestrator Agent — coordinates all sub-agents via Redis pub/sub.
"""

import json
import time
import threading
import redis
from typing import Optional

from .message_schemas import (
    AgentID, MessageType,
    PoseEstimateMessage, SituationVectorMessage,
    ActionRecommendationMessage, HumanOverrideMessage,
    ConsensusActionMessage, EscalationMessage,
    SystemStatusMessage, ConfidenceLevel
)
from .state_manager import StateManager
from .consensus import ConsensusEngine
from .armstrong_protocol import ArmstrongProtocol


from .redis_fallback import get_redis_client

CH_PERCEPTION   = "perception.out"
CH_COGNITION    = "cognition.out"
CH_ACTION       = "action.out"
CH_INTERFACE    = "interface.out"
CH_HUMAN_IN     = "human.in"
CH_CONSENSUS    = "orchestrator.consensus"
CH_ESCALATION   = "orchestrator.escalation"
CH_STATUS       = "orchestrator.status"
CH_HEARTBEAT    = "orchestrator.heartbeat"

ALL_CHANNELS = [
    CH_PERCEPTION, CH_COGNITION, CH_ACTION,
    CH_INTERFACE, CH_HUMAN_IN
]


class Orchestrator:
    DECISION_CYCLE_S = 1.0
    STATUS_CYCLE_S   = 5.0

    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 decision_timeout_s: int = 30,
                 hdc_layer=None):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.hdc_layer = hdc_layer

        self.redis_pub  = get_redis_client(host=redis_host, port=redis_port, db=0)
        self.redis_sub  = get_redis_client(host=redis_host, port=redis_port, db=0)

        self.state      = StateManager()
        self.consensus  = ConsensusEngine()
        self.armstrong  = ArmstrongProtocol(
            timeout_s=decision_timeout_s,
            on_timeout=self._on_armstrong_timeout,
            on_override=self._on_human_override
        )

        self._latest_perception: Optional[PoseEstimateMessage] = None
        self._latest_cognition:  Optional[SituationVectorMessage] = None
        self._latest_action:     Optional[ActionRecommendationMessage] = None
        self._latest_human:      Optional[HumanOverrideMessage] = None
        self._msg_lock = threading.Lock()

        self._running   = False
        self._cycle_count = 0

        print(f"Orchestrator initialized")
        print(f"  Redis: {redis_host}:{redis_port}")
        print(f"  Decision cycle: {self.DECISION_CYCLE_S}s")
        print(f"  Armstrong timeout: {decision_timeout_s}s")

    def start(self):
        self._running = True

        self._sub_thread = threading.Thread(
            target=self._subscriber_loop, daemon=True, name="orchestrator-subscriber"
        )
        self._sub_thread.start()

        self._decision_thread = threading.Thread(
            target=self._decision_loop, daemon=True, name="orchestrator-decision"
        )
        self._decision_thread.start()

        self._status_thread = threading.Thread(
            target=self._status_loop, daemon=True, name="orchestrator-status"
        )
        self._status_thread.start()

        print("Orchestrator started — all threads running")

    def stop(self):
        self._running = False
        print("Orchestrator stopped")

    def _subscriber_loop(self):
        pubsub = self.redis_sub.pubsub()
        pubsub.subscribe(*ALL_CHANNELS)
        print(f"Subscribed to channels: {ALL_CHANNELS}")

        for raw_msg in pubsub.listen():
            if not self._running:
                break
            if raw_msg["type"] != "message":
                continue

            channel = raw_msg["channel"].decode() if isinstance(raw_msg["channel"], bytes) else raw_msg["channel"]
            data    = raw_msg["data"].decode() if isinstance(raw_msg["data"], bytes) else raw_msg["data"]

            try:
                self._route_message(channel, data)
            except Exception as e:
                print(f"[Orchestrator] Error routing message from {channel}: {e}")

    def _route_message(self, channel: str, data: str):
        """
        Parse and route incoming message to correct handler.

        Uses Message.from_dict(payload) instead of Message(**payload) so
        transport-only keys (e.g. 'source': 'simulation' / 'real_model')
        added by publishers never crash construction. This was previously
        silently eating EVERY perception/cognition/action message via the
        broad except in _subscriber_loop, meaning consensus always ran on
        stale/default state.
        """
        payload = json.loads(data)

        with self._msg_lock:
            if channel == CH_PERCEPTION:
                self._latest_perception = PoseEstimateMessage.from_dict(payload)
                self.state.update_from_perception(self._latest_perception)
                print(f"[Orchestrator] Perception: "
                      f"confidence={self._latest_perception.confidence_level} "
                      f"JG={self._latest_perception.jensen_gain:.1f}°")

            elif channel == CH_COGNITION:
                self._latest_cognition = SituationVectorMessage.from_dict(payload)
                self.state.update_from_cognition(self._latest_cognition)
                print(f"[Orchestrator] Cognition: "
                      f"anomaly={self._latest_cognition.anomaly_detected} "
                      f"novelty={self._latest_cognition.novelty_score:.2f}")

            elif channel == CH_ACTION:
                self._latest_action = ActionRecommendationMessage.from_dict(payload)
                self.state.update_from_action(self._latest_action)
                print(f"[Orchestrator] Action: {self._latest_action.primary_action}")

            elif channel == CH_HUMAN_IN:
                self._latest_human = HumanOverrideMessage.from_dict(payload)
                self.state.update_from_human(self._latest_human)
                self.armstrong.receive_override(self._latest_human)
                print(f"[Orchestrator] Human override: "
                      f"Level {self._latest_human.override_level} "
                      f"-> {self._latest_human.selected_action}")

    def _decision_loop(self):
        while self._running:
            cycle_start = time.time()

            with self._msg_lock:
                p = self._latest_perception
                c = self._latest_cognition
                a = self._latest_action
                h = self._latest_human
                self._latest_human = None

            result = self.consensus.run(
                state=self.state.get_state(),
                perception_msg=p,
                cognition_msg=c,
                action_msg=a,
                human_msg=h
            )

            self.state.record_decision(
                action=result.final_action,
                reasoning=result.reasoning,
                consensus=result.consensus_reached,
                override=result.override_applied
            )

            self._publish(CH_CONSENSUS, result.to_json())

            if result.escalated_to_human and not result.override_applied:
                self._escalate(result)

            self._cycle_count += 1

            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.DECISION_CYCLE_S - elapsed)
            time.sleep(sleep_time)

    def _escalate(self, result: ConsensusActionMessage):
        esc = EscalationMessage(
            reason=result.reasoning,
            urgency=ConfidenceLevel.MODERATE,
            suggested_action=result.final_action,
            timeout_seconds=self.armstrong.timeout_s,
            context={
                "consensus_reached": result.consensus_reached,
                "votes": result.votes,
                "fallback": result.fallback_triggered
            }
        )
        self._publish(CH_ESCALATION, esc.to_json())

    def _status_loop(self):
        while self._running:
            time.sleep(self.STATUS_CYCLE_S)
            self.state.check_agent_health(timeout_s=10.0)
            s = self.state.get_state()

            status = SystemStatusMessage(
                perception_alive=s.agent_alive.get("perception", False),
                cognition_alive=s.agent_alive.get("cognition", False),
                action_alive=s.agent_alive.get("action", False),
                interface_alive=s.agent_alive.get("interface", False),
                overall_status="nominal" if all([
                    s.agent_alive.get("perception", False),
                    s.agent_alive.get("cognition", False),
                ]) else "degraded",
                cycle_time_ms=self.DECISION_CYCLE_S * 1000,
                total_cycles=self._cycle_count
            )
            self._publish(CH_STATUS, status.to_json())

    def _publish(self, channel: str, message: str):
        try:
            self.redis_pub.publish(channel, message)
        except Exception as e:
            print(f"[Orchestrator] Publish error on {channel}: {e}")

    def _on_armstrong_timeout(self):
        print("[Orchestrator] Armstrong timeout — publishing HOLD_POSITION")

    def _on_human_override(self, msg: HumanOverrideMessage):
        print(f"[Orchestrator] Override logged: {msg.override_level}")

        if self.hdc_layer is not None:
            try:
                latest_cog = self._latest_cognition
                if latest_cog and hasattr(latest_cog, 'situation_id'):
                    sit_b64 = getattr(latest_cog, 'situation_vector_b64', None)
                    if sit_b64:
                        self.hdc_layer.learn_outcome(
                            situation_vector_b64=sit_b64,
                            action_taken=str(msg.selected_action),
                            outcome="human_override",
                            success_rate=100.0,
                            metadata={
                                "override_level": str(msg.override_level),
                                "rationale": msg.rationale,
                                "operator": getattr(msg, 'operator_id', 'unknown')
                            }
                        )
                        print(f"[Orchestrator] HDC learned from override: {msg.selected_action}")
            except Exception as e:
                print(f"[Orchestrator] HDC learn_outcome failed: {e}")

    def publish_test_perception(self, confidence: str = "high",
                                jensen_gain: float = 1.0):
        from orchestrator.message_schemas import ActionType
        msg = PoseEstimateMessage(
            R=[[1,0,0],[0,1,0],[0,0,1]],
            t=[0.0, 0.0, 10.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            jensen_gain=jensen_gain,
            confidence_level=confidence,
            confidence_label=f"{confidence.upper()} CONFIDENCE",
            sigma_R_deg=jensen_gain * 0.6,
            sigma_t_m=0.1,
            nearest_anchor_idx=0,
            anchor_distance_deg=5.0,
            is_trustworthy=confidence in ("high", "moderate"),
            processing_time_ms=33.0,
            image_shape=[64, 64, 3]
        )
        self._publish(CH_PERCEPTION, msg.to_json())
        return msg
