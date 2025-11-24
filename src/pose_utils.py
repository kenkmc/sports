from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping, Optional

try:
    from mediapipe.solutions.pose import PoseLandmark
except ModuleNotFoundError:  # mediapipe >=0.10 packaged differently
    from mediapipe.python.solutions.pose import PoseLandmark

Keypoint = Mapping[str, float]
KeypointDict = Dict[str, Dict[str, float]]


def keypoints_to_dict(keypoints: Iterable[tuple[float, float, float, float]]) -> KeypointDict:
    """Convert the flat Mediapipe keypoint list into a dict keyed by landmark name."""
    kp_dict: KeypointDict = {}
    for idx, (x, y, z, visibility) in enumerate(keypoints):
        name = PoseLandmark(idx).name
        kp_dict[name] = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "visibility": float(visibility),
        }
    return kp_dict


def average_visibility(kp_dict: Mapping[str, Mapping[str, float]], required: Optional[Iterable[str]] = None) -> float:
    """Compute the average visibility across the provided landmarks."""
    if required:
        vals = [kp_dict[name]["visibility"] for name in required if name in kp_dict]
    else:
        vals = [v["visibility"] for v in kp_dict.values()]
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


class KeypointSmoother:
    """Applies an exponential moving average over keypoint coordinates."""

    def __init__(self, alpha: float = 0.3, visibility_alpha: float = 0.1):
        self.alpha = max(0.0, min(alpha, 1.0))
        self.visibility_alpha = max(0.0, min(visibility_alpha, 1.0))
        self._state: MutableMapping[str, Dict[str, float]] = {}

    def reset(self) -> None:
        self._state.clear()

    def smooth(self, kp_dict: KeypointDict) -> KeypointDict:
        smoothed: KeypointDict = {}
        for name, values in kp_dict.items():
            prev = self._state.get(name)
            if prev is None:
                smoothed_val = dict(values)
            else:
                smoothed_val = {
                    "x": prev["x"] + self.alpha * (values["x"] - prev["x"]),
                    "y": prev["y"] + self.alpha * (values["y"] - prev["y"]),
                    "z": prev["z"] + self.alpha * (values["z"] - prev["z"]),
                    "visibility": prev["visibility"]
                    + self.visibility_alpha * (values["visibility"] - prev["visibility"]),
                }
            self._state[name] = smoothed_val
            smoothed[name] = smoothed_val
        return smoothed


def get_avg(kp_dict: Mapping[str, Mapping[str, float]], names: Iterable[str], *, axis: str = "y", visibility_threshold: float = 0.5) -> Optional[float]:
    values = []
    for name in names:
        data = kp_dict.get(name)
        if not data or data.get("visibility", 0.0) < visibility_threshold:
            return None
        values.append(data[axis])
    if not values:
        return None
    return float(sum(values) / len(values))
