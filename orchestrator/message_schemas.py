from dataclasses import dataclass, asdict, field, fields
from typing import Optional
from enum import Enum
import json
import time


class AgentID(str, Enum):
    PERCEPTION  = "perception"
    COGNITION   = "cognition"
    ACTION      = "action"
    INTERFACE   = "interface"
    ORCHESTRATOR = "orchestrator"
    HUMAN       = "human"


class MessageType(str, Enum):
    POSE_ESTIMATE       = "pose_estimate"
    SITUATION_VECTOR    = "situation_vector"
    ANOMALY_DETECTED    = "anomaly_detected"
    ACTION_RECOMMENDATION = "action_recommendation"
    DISPLAY_UPDATE      = "display_update"
    HUMAN_OVERRIDE      = "human_override"
    HUMAN_ACKNOWLEDGE   = "human_acknowledge"
    CONSENSUS_ACTION    = "consensus_action"
    ESCALATION          = "escalation"
    HEARTBEAT           = "heartbeat"
    SYSTEM_STATUS       = "system_status"


class ConfidenceLevel(str, Enum):
    HIGH     = "high"
    MODERATE = "moderate"
    LOW      = "low"
    CRITICAL = "critical"


class ActionType(str, Enum):
    ABORT           = "abort"
    HOLD_POSITION   = "hold_position"
    PROCEED_SLOW    = "proceed_slow"
    PROCEED_NORMAL  = "proceed_normal"
    RECONFIGURE_POWER   = "reconfigure_power"
    ISOLATE_MODULE      = "isolate_module"
    EMERGENCY_VENT      = "emergency_vent"
    AWAIT_HUMAN         = "await_human"
    AUTONOMOUS_FALLBACK = "autonomous_fallback"


class OverrideLevel(str, Enum):
    ACKNOWLEDGE = "acknowledge"
    MODIFY      = "modify"
    REPLACE     = "replace"
    REJECT      = "reject"


@dataclass
class BaseMessage:
    agent_id:     str
    message_type: str
    timestamp:    float = field(default_factory=time.time)
    message_id:   str = field(default_factory=lambda: f"{time.time_ns()}")

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, data: dict):
        """
        Tolerant constructor: silently drops any keys that aren't real
        dataclass fields on `cls` (e.g. transport-only metadata like
        'source' or 'raw' added by publishers) instead of raising
        TypeError: unexpected keyword argument.

        This is the fix for the orchestrator swallowing every simulated
        and real-model message via a bare `except Exception: print(...)`.
        """
        if not isinstance(data, dict):
            raise TypeError(f"from_dict expects a dict, got {type(data)}")
        valid_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_names}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_str: str):
        return cls.from_dict(json.loads(json_str))


@dataclass
class PoseEstimateMessage(BaseMessage):
    agent_id:     str = AgentID.PERCEPTION
    message_type: str = MessageType.POSE_ESTIMATE

    R:          list = field(default_factory=list)
    t:          list = field(default_factory=list)
    quaternion: list = field(default_factory=list)

    jensen_gain:        float = 0.0
    confidence_level:   str   = ConfidenceLevel.HIGH
    confidence_label:   str   = ""
    sigma_R_deg:        float = 0.0
    sigma_t_m:          float = 0.0
    nearest_anchor_idx: int   = 0
    anchor_distance_deg: float = 0.0
    is_trustworthy:     bool  = True
    physics_residual_m: float = 0.0    
    physics_consistent: bool  = True

    processing_time_ms: float = 0.0
    image_shape:        list  = field(default_factory=list)


@dataclass
class SituationVectorMessage(BaseMessage):
    agent_id:     str = AgentID.COGNITION
    message_type: str = MessageType.SITUATION_VECTOR

    situation_id:       str   = ""
    anomaly_detected:   bool  = False
    anomaly_type:       str   = "none"
    anomaly_severity:   str   = ConfidenceLevel.HIGH
    novelty_score:      float = 0.0
    similar_case_id:    str   = ""
    similar_case_outcome: str = ""
    recommended_action: str   = ActionType.HOLD_POSITION
    action_confidence:  float = 0.0
    explanation:        str   = ""


@dataclass
class ActionRecommendationMessage(BaseMessage):
    agent_id:     str = AgentID.ACTION
    message_type: str = MessageType.ACTION_RECOMMENDATION

    primary_action:     str   = ActionType.HOLD_POSITION
    primary_score:      float = 0.0
    collision_prob:     float = 0.0
    collision_prob_upper_bound_99: float = 0.0
    mission_success_prob: float = 0.0
    resource_cost:      float = 0.0

    alternatives:       list  = field(default_factory=list)

    simulation_horizon_s: int   = 60
    mc_runs:              int   = 0
    explanation:          str   = ""


@dataclass
class HumanOverrideMessage(BaseMessage):
    agent_id:     str = AgentID.HUMAN
    message_type: str = MessageType.HUMAN_OVERRIDE

    override_level:    str = OverrideLevel.ACKNOWLEDGE
    selected_action:   str = ActionType.HOLD_POSITION
    rationale:         str = ""
    modified_params:   dict = field(default_factory=dict)
    operator_id:       str = "crew_commander"


@dataclass
class ConsensusActionMessage(BaseMessage):
    agent_id:     str = AgentID.ORCHESTRATOR
    message_type: str = MessageType.CONSENSUS_ACTION

    final_action:       str   = ActionType.HOLD_POSITION
    consensus_reached:  bool  = False
    votes:              dict  = field(default_factory=dict)
    override_applied:   bool  = False
    override_level:     str   = ""
    escalated_to_human: bool  = False
    reasoning:          str   = ""
    fallback_triggered: bool  = False


@dataclass
class EscalationMessage(BaseMessage):
    agent_id:     str = AgentID.ORCHESTRATOR
    message_type: str = MessageType.ESCALATION

    reason:           str   = ""
    urgency:          str   = ConfidenceLevel.MODERATE
    suggested_action: str   = ActionType.HOLD_POSITION
    timeout_seconds:  int   = 30
    context:          dict  = field(default_factory=dict)


@dataclass
class SystemStatusMessage(BaseMessage):
    agent_id:     str = AgentID.ORCHESTRATOR
    message_type: str = MessageType.SYSTEM_STATUS

    perception_alive:  bool  = False
    cognition_alive:   bool  = False
    action_alive:      bool  = False
    interface_alive:   bool  = False
    overall_status:    str   = "nominal"
    cycle_time_ms:     float = 0.0
    total_cycles:      int   = 0
