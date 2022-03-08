import tqdm
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
import csv

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

from embedder import FullBodyPoseEmbedder
from classifier import PoseClassifier, PoseSample, EMADictSmoothing
from counter import RepetitionCounter
from visualizer import PoseClassificationVisualizer


def get_pose_samples(pose_embedder, pose_samples_folder, 
                    n_landmarks=33, n_dimensions=3, 
                    file_extension="csv", file_separator=","):
    """Loads pose samples from a given folder.

    Required folder structure:
        neutral_standing.csv
        pushups_down.csv
        pushups_up.csv
        squats_down.csv
        ...

    Required CSV structure:
        sample_00001,x1,y1,z1,x2,y2,z2,....
        sample_00002,x1,y1,z1,x2,y2,z2,....
        ...
    """
    # Each file in the folder represents one pose class.
    file_names = [name for name in os.listdir(
        pose_samples_folder) if name.endswith(file_extension)]

    pose_samples = []
    for file_name in file_names:
        # Use file name as pose class name.
        class_name = file_name[:-(len(file_extension) + 1)]

        # Parse CSV.
        with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=file_separator)
            for row in csv_reader:
                assert len(row) == n_landmarks * n_dimensions + \
                    1, 'Wrong number of values: {}'.format(len(row))
                landmarks = np.array(row[1:], np.float32).reshape(
                    [n_landmarks, n_dimensions])
                pose_samples.append(PoseSample(
                    name=row[0],
                    landmarks=landmarks,
                    class_name=class_name,
                    embedding=pose_embedder(landmarks),
                ))

    return pose_samples


# Specify your video name and target pose class to count the repetitions.
video_path = 0
class_name = 'shoulderpress_down'
reference_video_path = 'data/shoulderpress-sample-out.mp4'

# Open the video.
video_cap = cv2.VideoCapture(video_path)

# Get some video parameters to generate output video with classificaiton.
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize tracker.
pose_tracker = mp_pose.Pose()

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Initalize samples
pose_samples_folder = 'models/shoulderpress'
pose_samples = get_pose_samples(pose_embedder, pose_samples_folder)

# Initialize classifier.
# Ceck that you are using the same parameters as during bootstrapping.
pose_classifier = PoseClassifier(
    pose_samples=pose_samples,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# Initialize EMA smoothing.
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# Initialize counter.
repetition_counter = RepetitionCounter(
    class_name=class_name,  # 预备动作类别
    enter_threshold=6,
    exit_threshold=4)

# Initialize renderer.
pose_classification_visualizer = PoseClassificationVisualizer(
    class_name=class_name,
    plot_x_max=video_n_frames,
    # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
    plot_y_max=10, 
    counter_font_path="data\Roboto-Regular.ttf"
    )


while True:
    # Get next frame of the video.
    success, input_frame = video_cap.read()
    if not success:
        break

    # Run pose tracker.
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
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

    pose_classification = pose_classifier(pose_landmarks)

    # Smooth classification using EMA.
    pose_classification_filtered = pose_classification_filter(pose_classification)

    # print(pose_classification_filtered)

    # Count repetitions.
    repetitions_count = repetition_counter(pose_classification_filtered)

    # Draw classification plot and repetition counter.
    output_frame = pose_classification_visualizer(
        frame=output_frame,
        pose_classification=pose_classification,
        pose_classification_filtered=pose_classification_filtered,
        repetitions_count=repetitions_count)

    cv2.imshow('frame', cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) &0xFF ==ord('q'):  #按q键退出
    	break

video_cap.release()
cv2.destroyAllWindows()

