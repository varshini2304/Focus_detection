from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, List
import time

# Sliding window of the last 30 minutes sampled per second
_window_seconds = 30 * 60
_samples = deque(maxlen=_window_seconds)
_last_sample_time: float | None = None

# Simple weighting for events
_EVENT_WEIGHT = {
    "phone": 1.0,
    "eyes_closed": 1.0,
    "yawn": 0.8,
    "talking": 0.5,
    "looking_away": 0.6,
}


def _fill_time_gaps(now: float):
    global _last_sample_time
    if _last_sample_time is None:
        _last_sample_time = now
        return
    # Add neutral samples if large time gaps occur between calls
    while _last_sample_time + 1.0 < now and len(_samples) < _window_seconds:
        _samples.append(0.0)
        _last_sample_time += 1.0
    _last_sample_time = now


async def update_and_compute(features: Dict[str, Any], events: List[str]) -> float:
    now = time.time()
    _fill_time_gaps(now)

    # Weighted distraction: clamp to 1 if any heavy event present
    distracted = 1.0 if any(e in _EVENT_WEIGHT for e in events) else 0.0
    _samples.append(distracted)

    if len(_samples) == 0:
        return 100.0

    distraction_ratio = sum(_samples) / len(_samples)
    focus_percent = max(0.0, 100.0 * (1.0 - distraction_ratio))
    return float(focus_percent)
