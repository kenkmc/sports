from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Mapping, Optional

from .pose_utils import get_avg


@dataclass
class SportState:
    name: str
    phase: str = "idle"
    count: int = 0
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=180))
    last_value: Optional[float] = None
    status: str = "calibrating"


class SportCounter:
    def __init__(
        self,
        name: str,
        metric_names: Iterable[str],
        *,
        high_when_down: bool,
        down_ratio: float,
        up_ratio: float,
        min_amplitude: float,
        warmup: int = 15,
    ) -> None:
        self.metric_names = tuple(metric_names)
        self.state = SportState(name=name)
        self.high_when_down = high_when_down
        self.down_ratio = down_ratio
        self.up_ratio = up_ratio
        self.min_amplitude = min_amplitude
        self.warmup = warmup

    def reset(self) -> None:
        self.state = SportState(name=self.state.name)

    def _calc_metric(self, keypoints: Mapping[str, Mapping[str, float]]) -> Optional[float]:
        return get_avg(keypoints, self.metric_names, axis="y")

    def update(self, keypoints: Mapping[str, Mapping[str, float]]) -> SportState:
        value = self._calc_metric(keypoints)
        if value is None:
            self.state.status = "waiting"
            return self.state

        self.state.history.append(value)
        self.state.last_value = value

        if len(self.state.history) < self.warmup:
            self.state.status = "calibrating"
            return self.state

        mn = min(self.state.history)
        mx = max(self.state.history)
        amplitude = mx - mn
        if amplitude < self.min_amplitude:
            self.state.status = "too_still"
            return self.state

        enter_threshold = mn + self.down_ratio * amplitude
        exit_threshold = mn + self.up_ratio * amplitude

        if self.high_when_down:
            if self.state.phase in ("idle", "up") and value > enter_threshold:
                self.state.phase = "down"
                self.state.status = "down"
            elif self.state.phase == "down" and value < exit_threshold:
                self.state.phase = "up"
                self.state.count += 1
                self.state.status = "rep"
            else:
                self.state.status = self.state.phase or "up"
        else:
            # For movements where metric decreases during the active phase (e.g. jump where feet lift)
            if self.state.phase in ("idle", "ground") and value < enter_threshold:
                self.state.phase = "air"
                self.state.status = "air"
            elif self.state.phase == "air" and value > exit_threshold:
                self.state.phase = "ground"
                self.state.count += 1
                self.state.status = "rep"
            else:
                self.state.status = self.state.phase or "ground"

        if self.state.phase == "idle":
            self.state.phase = "up" if self.high_when_down else "ground"
        return self.state


class RepCounter:
    def __init__(self) -> None:
        self._counters: Dict[str, SportCounter] = {
            "pushup": SportCounter(
                "pushup",
                ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"),
                high_when_down=True,
                down_ratio=0.70,
                up_ratio=0.45,
                min_amplitude=0.05,
                warmup=20,
            ),
            "squat": SportCounter(
                "squat",
                ("LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"),
                high_when_down=True,
                down_ratio=0.65,
                up_ratio=0.40,
                min_amplitude=0.04,
                warmup=20,
            ),
            "jump": SportCounter(
                "jump",
                ("LEFT_ANKLE", "RIGHT_ANKLE"),
                high_when_down=False,
                down_ratio=0.30,
                up_ratio=0.75,
                min_amplitude=0.05,
                warmup=25,
            ),
            "jumping_jack": SportCounter(
                "jumping_jack",
                ("LEFT_WRIST", "RIGHT_WRIST"),
                high_when_down=False,
                down_ratio=0.30,
                up_ratio=0.75,
                min_amplitude=0.15,
                warmup=20,
            ),
            "lunge": SportCounter(
                "lunge",
                ("LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE"),
                high_when_down=True,
                down_ratio=0.65,
                up_ratio=0.40,
                min_amplitude=0.04,
                warmup=20,
            ),
        }

    def reset(self) -> None:
        for counter in self._counters.values():
            counter.reset()

    def snapshot(self, sport: str) -> Dict[str, object]:
        counter = self._counters.get(sport)
        payload = self._serialize(counter.state if counter else None, sport)
        payload["totals"] = self._totals()
        return payload

    def update(
        self,
        sport: str,
        keypoints: Mapping[str, Mapping[str, float]],
        *,
        confidence: float,
    ) -> Dict[str, object]:
        counter = self._counters.get(sport)
        if not counter:
            return {
                "sport": sport,
                "count": 0,
                "phase": "idle",
                "status": "unsupported",
                "confidence": confidence,
                "totals": self._totals(),
            }

        if confidence < 0.35:
            # Don't update counts when confidence is too low
            snapshot = self._serialize(counter.state, sport)
            snapshot.update({"confidence": confidence, "status": "low_confidence"})
            return snapshot

        state = counter.update(keypoints)
        payload = self._serialize(state, sport)
        payload["confidence"] = confidence
        payload["totals"] = self._totals()
        return payload

    def _serialize(self, state: Optional[SportState], sport: str) -> Dict[str, object]:
        if state is None:
            return {
                "sport": sport,
                "count": 0,
                "phase": "idle",
                "status": "no_data",
            }
        return {
            "sport": sport,
            "count": state.count,
            "phase": state.phase,
            "status": state.status,
            "metric": state.last_value,
        }

    def _totals(self) -> Dict[str, int]:
        return {name: counter.state.count for name, counter in self._counters.items()}
