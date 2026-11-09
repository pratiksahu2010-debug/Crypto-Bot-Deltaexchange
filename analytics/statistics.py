"""Summary statistics helpers."""
from typing import List
import statistics

def mean_score(scores: List[float]) -> float:
    return statistics.mean(scores) if scores else 0.0
