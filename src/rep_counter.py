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
        range_window: int = 75,
        transition_frames: int = 2,
        metric_alpha: float = 0.35,
    ) -> None:
        self.metric_names = tuple(metric_names)
        self.state = SportState(name=name)
        self.high_when_down = high_when_down
        self.down_ratio = down_ratio
        self.up_ratio = up_ratio
        self.min_amplitude = min_amplitude
        self.warmup = warmup
        self.range_window = max(10, int(range_window))
        self.transition_frames = max(1, int(transition_frames))
        self.metric_alpha = max(0.0, min(float(metric_alpha), 1.0))
        self._enter_streak = 0
        self._exit_streak = 0
        self._ema_value: Optional[float] = None

    def reset(self) -> None:
        self.state = SportState(name=self.state.name)
        self._enter_streak = 0
        self._exit_streak = 0
        self._ema_value = None

    def _calc_metric(self, keypoints: Mapping[str, Mapping[str, float]]) -> Optional[float]:
        return get_avg(keypoints, self.metric_names, axis="y")

    def update(self, keypoints: Mapping[str, Mapping[str, float]]) -> SportState:
        raw_value = self._calc_metric(keypoints)
        if raw_value is None:
            self.state.status = "waiting"
            self._enter_streak = 0
            self._exit_streak = 0
            return self.state

        if self._ema_value is None:
            value = raw_value
        else:
            value = self._ema_value + self.metric_alpha * (raw_value - self._ema_value)
        self._ema_value = value

        self.state.history.append(value)
        self.state.last_value = value

        if len(self.state.history) < self.warmup:
            self.state.status = "calibrating"
            return self.state

        recent = list(self.state.history)[-self.range_window:]
        mn = min(recent)
        mx = max(recent)
        amplitude = mx - mn
        if amplitude < self.min_amplitude:
            self.state.status = "too_still"
            self._enter_streak = 0
            self._exit_streak = 0
            return self.state

        enter_threshold = mn + self.down_ratio * amplitude
        exit_threshold = mn + self.up_ratio * amplitude

        if self.high_when_down:
            if self.state.phase in ("idle", "up"):
                if value > enter_threshold:
                    self._enter_streak += 1
                else:
                    self._enter_streak = 0
                if self._enter_streak >= self.transition_frames:
                    self.state.phase = "down"
                    self.state.status = "down"
                    self._enter_streak = 0
                    self._exit_streak = 0
                else:
                    self.state.status = self.state.phase or "up"
            elif self.state.phase == "down":
                if value < exit_threshold:
                    self._exit_streak += 1
                else:
                    self._exit_streak = 0
                if self._exit_streak >= self.transition_frames:
                    self.state.phase = "up"
                    self.state.count += 1
                    self.state.status = "rep"
                    self._enter_streak = 0
                    self._exit_streak = 0
                else:
                    self.state.status = "down"
            else:
                self.state.status = self.state.phase or "up"
        else:
            # For movements where metric decreases during the active phase (e.g. jump where feet lift)
            if self.state.phase in ("idle", "ground"):
                if value < enter_threshold:
                    self._enter_streak += 1
                else:
                    self._enter_streak = 0
                if self._enter_streak >= self.transition_frames:
                    self.state.phase = "air"
                    self.state.status = "air"
                    self._enter_streak = 0
                    self._exit_streak = 0
                else:
                    self.state.status = self.state.phase or "ground"
            elif self.state.phase == "air":
                if value > exit_threshold:
                    self._exit_streak += 1
                else:
                    self._exit_streak = 0
                if self._exit_streak >= self.transition_frames:
                    self.state.phase = "ground"
                    self.state.count += 1
                    self.state.status = "rep"
                    self._enter_streak = 0
                    self._exit_streak = 0
                else:
                    self.state.status = "air"
            else:
                self.state.status = self.state.phase or "ground"

        if self.state.phase == "idle":
            self.state.phase = "up" if self.high_when_down else "ground"
        return self.state


class RepCounter:
    def __init__(self, *, confidence_threshold: float = 0.30) -> None:
        self.confidence_threshold = max(0.0, min(float(confidence_threshold), 1.0))
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
                transition_frames=1,
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

    def required_landmarks(self, sport: str) -> tuple[str, ...]:
        counter = self._counters.get(sport)
        if not counter:
            return ()
        return counter.metric_names

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

        if confidence < self.confidence_threshold:
            # Don't update counts when confidence is too low
            snapshot = self._serialize(counter.state, sport)
            snapshot.update({
                "confidence": confidence,
                "status": "low_confidence",
                "totals": self._totals(),
            })
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
