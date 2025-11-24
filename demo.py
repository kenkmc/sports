"""A lightweight demo that runs the pose detector and simple tracker on webcam frames.

Requirements: installed packages from requirements.txt, a working webcam.
"""

import os
import time
import argparse
import cv2
import numpy as np
# Compatibility shim: some libraries expect google.protobuf.message_factory.MessageFactory
# to have a GetPrototype method (older API). Newer protobuf versions may differ.
try:
    from google.protobuf import message_factory as _mf
    if not hasattr(_mf.MessageFactory, 'GetPrototype') and hasattr(_mf.MessageFactory, 'GetPrototype') == False:
        # If the method is missing, alias to existing CreatePrototype/GetPrototype variations
        def _get_prototype(self, descriptor):
            # some versions use CreatePrototype
            if hasattr(self, 'CreatePrototype'):
                return self.CreatePrototype(descriptor)
            raise AttributeError('MessageFactory has no GetPrototype or CreatePrototype')
        setattr(_mf.MessageFactory, 'GetPrototype', _get_prototype)
except Exception:
    # if protobuf isn't installed or different layout, ignore — demo will fail elsewhere if needed
    pass

from src.pose_detector import PoseDetector
from src.tracker import SimpleTracker
from src.sheets_uploader import SheetsUploader


def landmarks_to_bbox(landmarks, w, h, margin=0.05):
    # landmarks: list of (x,y,z,visibility) normalized
    xs = [l[0] for l in landmarks]
    ys = [l[1] for l in landmarks]
    if not xs or not ys:
        return None
    x_min = max(0.0, min(xs) - margin)
    y_min = max(0.0, min(ys) - margin)
    x_max = min(1.0, max(xs) + margin)
    y_max = min(1.0, max(ys) + margin)
    x = int(x_min * w)
    y = int(y_min * h)
    ww = int((x_max - x_min) * w)
    hh = int((y_max - y_min) * h)
    return (x, y, ww, hh)


def main(args):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('No camera found. Exiting.')
        return

    pd = PoseDetector(model_complexity=0, smooth=True)
    tracker = SimpleTracker(iou_threshold=args.iou_threshold)

    sheet_id = args.sheet_id or os.environ.get('SHEET_ID')
    creds_path = args.creds_path or os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    uploader = None
    if sheet_id and creds_path:
        uploader = SheetsUploader(sheet_id=sheet_id, creds_path=creds_path)

    try:
        cv2.namedWindow('demo', cv2.WINDOW_NORMAL)
        start_time = time.time()
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints, landmarks_obj = pd.detect(rgb)
            bb = None
            if keypoints:
                bb = landmarks_to_bbox(keypoints, w, h)
                detections = [bb]
            else:
                detections = []

            tracks = tracker.update(detections)

            # draw
            # draw bounding box
            if bb:
                x, y, ww, hh = bb
                cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            # draw tracks
            for tid, tbb in tracks:
                x, y, ww, hh = tbb
                cv2.rectangle(frame, (x, y), (x + ww, y + hh), (255, 128, 0), 2)
                cv2.putText(frame, str(tid), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # draw landmarks if available (use landmarks_obj for richer info)
            if landmarks_obj is not None:
                for lm in landmarks_obj.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

            # overlay info
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            info = f'FPS: {fps:.1f}  Size: {w}x{h}  Kpts: {len(keypoints)}'
            cv2.putText(frame, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            if not args.headless:
                cv2.imshow('demo', frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1
            if key == ord('q'):
                break

            if uploader and keypoints:
                rec = {'id': tracks[0][0] if tracks else '', 'time': time.time(), 'keypoints_count': len(keypoints)}
                uploader.upload_record(rec)

            # auto-stop if run-time provided
            if args.run_time and (time.time() - start_time) > args.run_time:
                print('Run-time exceeded, exiting')
                break

    finally:
        pd.close()
        cap.release()
        cv2.destroyAllWindows()


def _parse_args():
    p = argparse.ArgumentParser(description='Demo for pose detector + simple tracker')
    p.add_argument('--camera', type=int, default=0, help='camera index (default 0)')
    p.add_argument('--iou-threshold', type=float, default=0.3, help='IoU threshold for tracker')
    p.add_argument('--sheet-id', type=str, default=None, help='Google sheet id to upload records')
    p.add_argument('--creds-path', type=str, default=None, help='Path to service account JSON')
    p.add_argument('--headless', action='store_true', help='Run without opening a display window')
    p.add_argument('--run-time', type=float, default=0.0, help='Auto exit after N seconds (0 = no limit)')
    return p.parse_args()


if __name__ == '__main__':
    main(_parse_args())
