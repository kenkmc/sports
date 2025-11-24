import numpy as np


def iou(bb1, bb2):
    # bb = (x, y, w, h)
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    def __init__(self, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = {}  # id -> bb
        self.iou_threshold = iou_threshold

    def update(self, detections):
        # detections: list of bb (x,y,w,h)
        assigned = {}
        used = set()
        for det in detections:
            best_id = None
            best_iou = 0
            for tid, tbb in self.tracks.items():
                val = iou(det, tbb)
                if val > best_iou:
                    best_iou = val
                    best_id = tid
            if best_iou >= self.iou_threshold and best_id not in used:
                assigned[best_id] = det
                used.add(best_id)
            else:
                # new track
                assigned[self.next_id] = det
                self.next_id += 1

        # remove unmatched old tracks
        self.tracks = assigned.copy()
        return list(self.tracks.items())  # list of (id, bb)
