import pytest

try:
    from src.rep_counter import RepCounter
except Exception as exc:  # pragma: no cover - skip if mediapipe missing
    pytest.skip(f"mediapipe not available: {exc}", allow_module_level=True)


def make_pose(value_map):
    pose = {}
    for name, y in value_map.items():
        pose[name] = {"x": 0.5, "y": y, "z": 0.0, "visibility": 0.95}
    return pose


def cycle(rep_counter, sport, values, confidence=0.9):
    info = None
    for y in values:
        pose = make_pose({
            "LEFT_SHOULDER": y,
            "RIGHT_SHOULDER": y,
            "LEFT_HIP": y,
            "RIGHT_HIP": y,
            "LEFT_KNEE": y + 0.05,
            "RIGHT_KNEE": y + 0.05,
            "LEFT_ANKLE": y + 0.1,
            "RIGHT_ANKLE": y + 0.1,
        })
        info = rep_counter.update(sport, pose, confidence=confidence)
    return info


def test_pushup_counts_increment():
    rc = RepCounter()
    pattern = [0.35] * 20 + [0.68] * 20
    info = None
    for _ in range(4):
        info = cycle(rc, "pushup", pattern)
    assert info["count"] >= 3
    assert info["sport"] == "pushup"


def test_jump_requires_confidence():
    rc = RepCounter()
    pose = make_pose({
        "LEFT_ANKLE": 0.9,
        "RIGHT_ANKLE": 0.9,
        "LEFT_SHOULDER": 0.4,
        "RIGHT_SHOULDER": 0.4,
        "LEFT_HIP": 0.5,
        "RIGHT_HIP": 0.5,
    })
    # Low confidence should not change counts
    info = rc.update("jump", pose, confidence=0.2)
    assert info["count"] == 0
    assert info["status"] == "low_confidence"
    assert "totals" in info
    assert isinstance(info["totals"], dict)


def test_jumping_jack_counts():
    rc = RepCounter()
    # Arms down (0.8) -> Arms up (0.2) -> Arms down (0.8)
    # Warmup first
    warmup = [0.8] * 25
    cycle(rc, "jumping_jack", warmup)
    
    # Perform reps
    # Down (0.8) -> Up (0.2) -> Down (0.8)
    rep_pattern = [0.8] * 10 + [0.2] * 10 + [0.8] * 10
    
    info = None
    for _ in range(3):
        # We need to pass wrist coordinates
        for y in rep_pattern:
            pose = make_pose({
                "LEFT_WRIST": y,
                "RIGHT_WRIST": y,
                "LEFT_SHOULDER": 0.5,
                "RIGHT_SHOULDER": 0.5,
            })
            info = rc.update("jumping_jack", pose, confidence=0.9)
            
    assert info["count"] >= 3
    assert info["sport"] == "jumping_jack"


def test_lunge_counts():
    rc = RepCounter()
    # Standing (0.2) -> Lunge (0.8) -> Standing (0.2)
    warmup = [0.2] * 25
    cycle(rc, "lunge", warmup)
    
    rep_pattern = [0.2] * 10 + [0.8] * 10 + [0.2] * 10
    
    info = None
    for _ in range(3):
        for y in rep_pattern:
            pose = make_pose({
                "LEFT_HIP": y,
                "RIGHT_HIP": y,
                "LEFT_KNEE": y + 0.1,
                "RIGHT_KNEE": y + 0.1,
            })
            info = rc.update("lunge", pose, confidence=0.9)
            
    assert info["count"] >= 3
    assert info["sport"] == "lunge"
