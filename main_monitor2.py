from ast import For
import tqdm
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import pickle as pk
import csv

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from embedder import FullBodyPoseEmbedder


def cos_similarity(v1, v2):
    num = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom if denom != 0 else 0


def get_index_to_line(landmarks):
    index2vector = {}
    index2vector[0] = ((landmarks[2] + landmarks[5]) * 0.5, (landmarks[9] + landmarks[10]) * 0.5)  # 眼中到嘴中
    index2vector[1] = ((landmarks[23] + landmarks[24]) * 0.5, (landmarks[11] + landmarks[12]) * 0.5)  # 臀中到肩中
    index2vector[2] = (landmarks[11], landmarks[13])  # 左肩到左肘
    index2vector[3] = (landmarks[12], landmarks[14])  # 右肩到右肘
    index2vector[4] = (landmarks[13], landmarks[15])  # 左肘到左腕
    index2vector[5] = (landmarks[14], landmarks[16])  # 右肘到右腕
    index2vector[6] = (landmarks[23], landmarks[25])  # 左臀到左膝
    index2vector[7] = (landmarks[24], landmarks[26])  # 右臀到右膝
    index2vector[8] = (landmarks[25], landmarks[27])  # 左膝到左脚踝
    index2vector[9] = (landmarks[26], landmarks[28])  # 右膝到右脚踝

    index2vector[10] = (landmarks[11], landmarks[15])  # 左肩到左腕
    index2vector[11] = (landmarks[12], landmarks[16])  # 右肩到右腕
    index2vector[12] = (landmarks[23], landmarks[27])  # 左臀到左脚踝
    index2vector[13] = (landmarks[24], landmarks[28])  # 右臀到右脚踝

    index2vector[14] = (landmarks[23], landmarks[15])  # 左臀到左腕
    index2vector[15] = (landmarks[24], landmarks[16])  # 右臀到右腕
    
    index2vector[16] = (landmarks[11], landmarks[27])  # 左肩到左脚踝
    index2vector[17] = (landmarks[12], landmarks[28])  # 右肩到右脚踝
    index2vector[18] = (landmarks[23], landmarks[15])  # 左臀到左腕
    index2vector[19] = (landmarks[24], landmarks[16])  # 右臀到右腕

    index2vector[20] = (landmarks[13], landmarks[14])  # 左肘到右肘
    index2vector[21] = (landmarks[27], landmarks[28])  # 左膝到右膝
    index2vector[22] = (landmarks[15], landmarks[16])  # 左腕到右腕
    index2vector[23] = (landmarks[27], landmarks[28])  # 左脚踝到右脚踝

    return index2vector

# Specify your video name and target pose class to count the repetitions.
video_path = 0
class_name = 'shoulderpress_down'
reference_video_path = 'data/FullBody_Patch.mp4'
reference_datas_path = 'data/course01.pkl'

with open(reference_datas_path, mode='rb') as rf:
    refer_pose_datas = pk.load(rf)

ready_image_path = "data/sholderpress_ready.jpg"
ready_image = cv2.cvtColor(cv2.imread(ready_image_path), cv2.COLOR_BGR2RGB)

# Open the video.
reference_video_cap = cv2.VideoCapture(reference_video_path)
reference_video_cap.set(cv2.CAP_PROP_FPS, 25)
video_cap = cv2.VideoCapture(0)
video_cap.set(cv2.CAP_PROP_FPS, 25)

# Initialize tracker.
pose_tracker = mp_pose.Pose()

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Get ready embedding 
result = pose_tracker.process(image=ready_image)
ready_pose_landmarks = result.pose_landmarks
assert(ready_pose_landmarks is not None, "Please give the correct ready image!")

ready_pose_landmarks = np.array([[lmk.x * 640, lmk.y * 480, lmk.z * 640] for lmk in ready_pose_landmarks.landmark], dtype=np.float32)
ready_embedding = pose_embedder(ready_pose_landmarks)

is_start = False

while len(refer_pose_datas) > 0:
    # 1. 开始先遍历视频，直到与准备动作匹配
    while not is_start:
        success, input_frame = video_cap.read()
        if not success:
            break
        # Run pose tracker.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)[:, ::-1, :]
        result = pose_tracker.process(image=input_frame)
        pose_landmarks = result.pose_landmarks

        # Draw pose prediction.
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)

        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in pose_landmarks.landmark], dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

        mean_similar = 0.0
        if pose_landmarks is not None:
            pose_embedding = pose_embedder(pose_landmarks)
            similars = []
            for v1, v2 in zip(ready_embedding, pose_embedding):
                similar = cos_similarity(v1[:2], v2[:2])
                similars.append(similar)
            mean_similar = sum(similars) / len(similars)
        # print(similars)
        # print(mean_similar)
        # print("*" * 100)
        # exit()
        cv2.imshow('frame', cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) &0xFF ==ord('q'):  #按q键退出
            exit()

        if mean_similar > 0.99:
            is_start = True
            cv2.destroyAllWindows()
            break
    
    # 2. 循环直到参考视频结束为止，期间输入流取一帧
    refer_success, refer_input_frame = reference_video_cap.read()
    if not refer_success:
        break
    success, input_frame = video_cap.read()

    # Run pose tracker.
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)[:, ::-1, :]
    refer_input_frame = cv2.cvtColor(refer_input_frame, cv2.COLOR_BGR2RGB)

    result = pose_tracker.process(image=input_frame)

    pose_landmarks = result.pose_landmarks
    refer_pose_landmarks = refer_pose_datas.pop(0)

    # Draw pose prediction.
    output_frame = input_frame.copy()
    refer_output_frame = refer_input_frame.copy()
    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]

    if pose_landmarks is not None:
        # Embedding index to vector mapping 

        mp_drawing.draw_landmarks(
            image=output_frame,
            landmark_list=pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS)

    if refer_pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            image=refer_output_frame,
            landmark_list=refer_pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS)

    if pose_landmarks is not None:
        # Get landmarks.
        pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in pose_landmarks.landmark], dtype=np.float32)
        assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)
        index2vector = get_index_to_line(pose_landmarks)

    if refer_pose_landmarks is not None:
        # Get landmarks.
        refer_pose_landmarks = np.array([[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in refer_pose_landmarks.landmark], dtype=np.float32)
        assert refer_pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(refer_pose_landmarks.shape)

    mean_similar = 0.0
    similars = []
    if pose_landmarks is not None and refer_pose_landmarks is not None:
        pose_embedding = pose_embedder(pose_landmarks)
        refer_pose_embedding = pose_embedder(refer_pose_landmarks)

        for i, (v1, v2) in enumerate(zip(refer_pose_embedding, pose_embedding)):
            similar = cos_similarity(v1[:2], v2[:2])
            similars.append(similar)
            if similar < 0.8:
                cv2.line(output_frame, index2vector[i][0][:2].astype(np.int32), index2vector[i][1][:2].astype(np.int32), (255, 0, 0), 3)
                
        mean_similar = sum(similars) / (len(similars) + 1e-8)

    refer_output_frame = cv2.resize(refer_output_frame, (frame_width, frame_height))

    output_frame = np.hstack([output_frame, refer_output_frame])

    is_standard = False
    if mean_similar > 0.97:
        is_standard = True
    
    cv2.putText(output_frame, 'Is standard: {}'.format(str(is_standard)), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.imshow('frame', output_frame[:, :, ::-1])
    if cv2.waitKey(1) &0xFF ==ord('q'):
        break


reference_video_cap.release()
video_cap.release()
cv2.destroyAllWindows()



