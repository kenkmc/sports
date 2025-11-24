import pytest
from src.tracker import iou, SimpleTracker


def test_iou_non_overlapping():
    assert iou((0,0,10,10),(20,20,10,10)) == 0.0


def test_iou_partial_overlap():
    v = iou((0,0,10,10),(5,5,10,10))
    assert 0.0 < v < 1.0


def test_tracker_assign_new_ids():
    st = SimpleTracker(iou_threshold=0.3)
    dets1 = [(0,0,10,10),(100,100,10,10)]
    tracks = st.update(dets1)
    assert len(tracks) == 2
    ids = [t[0] for t in tracks]
    assert set(ids) == {1,2}


def test_tracker_match_existing():
    st = SimpleTracker(iou_threshold=0.1)
    dets1 = [(0,0,10,10)]
    tracks1 = st.update(dets1)
    assert tracks1[0][0] == 1
    # slightly moved detection
    dets2 = [(2,2,10,10)]
    tracks2 = st.update(dets2)
    assert tracks2[0][0] == 1
