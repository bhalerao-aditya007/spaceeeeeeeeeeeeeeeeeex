from dataclasses import dataclass
from typing import Dict, List, Optional

SUBSYSTEM_GRAPH: Dict[str, List[str]] = {
    "thermal": ["power"],
    "power": ["life_support"],
    "life_support": [],
}
SEVERITY_RANK = {"nominal": 0, "degraded": 1, "critical": 2, "failed": 3}


@dataclass
class SubsystemState:
    name: str
    severity: str


def find_root_cause(states: Dict[str, SubsystemState]) -> dict:
    critical_nodes = [name for name, s in states.items()
                       if SEVERITY_RANK.get(s.severity, 0) >= SEVERITY_RANK["critical"]]
    if not critical_nodes:
        return {"root_cause": None, "cascade": [], "narrative": "All subsystems nominal."}

    parents = {child: parent for parent, children in SUBSYSTEM_GRAPH.items() for child in children}
    roots = [n for n in critical_nodes if parents.get(n) not in critical_nodes]
    if not roots:
        roots = critical_nodes  # defensive: cycle-free graph guarantees this won't normally hit

    def downstream_chain(root):
        chain = [root]
        frontier = SUBSYSTEM_GRAPH.get(root, [])
        while frontier:
            nxt = [f for f in frontier if f in critical_nodes]
            if not nxt:
                break
            chain.extend(nxt)
            frontier = [g for f in nxt for g in SUBSYSTEM_GRAPH.get(f, [])]
        return chain

    chains = [downstream_chain(r) for r in roots]
    primary_chain = max(chains, key=len)
    narrative = " -> ".join(f"{node} ({states[node].severity})" for node in primary_chain)
    return {
        "root_cause": primary_chain[0],
        "cascade": primary_chain,
        "narrative": f"ROOT CAUSE: {narrative}. Address {primary_chain[0]}, not the downstream symptoms."
    }
