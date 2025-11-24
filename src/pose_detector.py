import cv2
import numpy as np

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except AttributeError:
    from mediapipe.python.solutions import pose as mp_pose


class PoseDetector:
    def __init__(self, model_complexity=0, smooth=True):
        self.pose = mp_pose.Pose(static_image_mode=False,
                                  model_complexity=model_complexity,
                                  smooth_landmarks=smooth,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

    def detect(self, image_rgb):
        # expects RGB image
        res = self.pose.process(image_rgb)
        keypoints = []
        if res.pose_landmarks:
            for lm in res.pose_landmarks.landmark:
                keypoints.append((lm.x, lm.y, lm.z, lm.visibility))
        return keypoints, res.pose_landmarks

    def close(self):
        self.pose.close()
