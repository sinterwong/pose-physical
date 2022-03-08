from ast import For
import tqdm
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import csv
import pickle
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from embedder import FullBodyPoseEmbedder


# Open the video.
video_path = 'data/FullBody_Patch.mp4'
out_path = 'data/course01.pkl'
video_cap = cv2.VideoCapture(video_path)
video_cap.set(cv2.CAP_PROP_FPS, 25)

pose_tracker = mp_pose.Pose()

results = []
while True:
    success, input_frame = video_cap.read()
    if not success:
        break

    # Run pose tracker.
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

    result = pose_tracker.process(image=input_frame)

    pose_landmarks = result.pose_landmarks
    results.append(pose_landmarks)

video_cap.release()

with open(out_path, mode='wb') as wf:
    pickle.dump(results, wf)

