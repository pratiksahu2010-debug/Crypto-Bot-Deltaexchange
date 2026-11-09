"""Basic performance metrics."""
from typing import List, Dict

def compute_hit_rates(records: List[dict]) -> Dict[str, float]:
    if not records:
        return {}
    total = len(records)
    return {
        "tp1_rate": sum(1 for r in records if r.get("tp1_hit")) / total,
        "sl_rate": sum(1 for r in records if r.get("sl_hit")) / total,
    }
