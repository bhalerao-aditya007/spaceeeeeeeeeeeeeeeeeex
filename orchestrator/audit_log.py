"""
Hash-chained, append-only decision log. Any post-hoc edit/delete/insert
breaks the chain and is detectable by HashChainedLog.verify().
"""
import hashlib
import json
import time
from pathlib import Path


class HashChainedLog:
    GENESIS_HASH = "0" * 64

    def __init__(self, log_path: str = "orchestrator/logs/decision_log.jsonl"):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = self._recover_last_hash()

    def _recover_last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return self.GENESIS_HASH
        with open(self.path, "rb") as f:
            lines = f.readlines()
        if not lines:
            return self.GENESIS_HASH
        return json.loads(lines[-1])["entry_hash"]

    def append(self, decision: dict) -> str:
        payload = {
            "timestamp": time.time(),
            "prev_hash": self._last_hash,
            "decision": decision,
        }
        payload_bytes = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        entry_hash = hashlib.sha256(payload_bytes).hexdigest()
        payload["entry_hash"] = entry_hash

        with open(self.path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")

        self._last_hash = entry_hash
        return entry_hash

    @staticmethod
    def verify(log_path: str) -> dict:
        prev = HashChainedLog.GENESIS_HASH
        n_verified = 0
        p = Path(log_path)
        if not p.exists():
            return {"valid": True, "entries_verified": 0, "note": "log does not exist yet"}
        with open(p) as f:
            for line_no, line in enumerate(f, 1):
                entry = json.loads(line)
                claimed_hash = entry.pop("entry_hash")
                recomputed = hashlib.sha256(
                    json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
                if entry["prev_hash"] != prev or recomputed != claimed_hash:
                    return {"valid": False, "broken_at_line": line_no}
                prev = claimed_hash
                n_verified += 1
        return {"valid": True, "entries_verified": n_verified}
