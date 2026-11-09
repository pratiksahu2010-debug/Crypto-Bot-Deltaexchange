"""Track signal outcomes for analytics."""
from typing import Dict, List
from collections import defaultdict

class OutcomeTracker:
    def __init__(self):
        self.records: List[dict] = []

    def record(self, signal: dict, outcome: str) -> None:
        self.records.append({"signal": signal, "outcome": outcome})

    def summary(self) -> dict:
        total = len(self.records)
        if not total:
            return {"total": 0}
        by = defaultdict(int)
        for r in self.records:
            by[r["outcome"]] += 1
        return {"total": total, "by_outcome": dict(by)}
