#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import csv
import copy
import numpy as np
import pyautogui

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from model import KeyPointClassifier

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "model/hand_landmarker.task"
LABEL_PATH = "model/keypoint_classifier/keypoint_classifier_label.csv"

pyautogui.FAILSAFE = False
ACTION_DELAY = 1.0  # seconds between key presses

# ----------------------------
# LOAD LABELS
# ----------------------------
with open(LABEL_PATH, encoding="utf-8-sig") as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

keypoint_classifier = KeyPointClassifier()

# ----------------------------
# MEDIAPIPE SETUP
# ----------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# ----------------------------
# VLC / MEDIA CONTROL
# ----------------------------
last_action_time = 0

def control_media(gesture_name):
    global last_action_time
    now = time.time()
    if now - last_action_time < ACTION_DELAY:
        return

    if gesture_name == "Open":
        pyautogui.press("space")       # Play / Pause
        print("[ACTION] Play / Pause")

    elif gesture_name == "Close":
        pyautogui.press("f")           # Full Screen
        print("[ACTION] FullScreen")

    elif gesture_name == "Thumbs Up":
        pyautogui.press("up")         # Volume up
        print("[ACTION] Volume Up")

    elif gesture_name == "Thumbs  Down":
        pyautogui.press("down")        # Volume down
        print("[ACTION] Volume Down")



    last_action_time = now

# ----------------------------
# LANDMARK UTILS
# ----------------------------
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for lm in landmarks:
        landmark_point.append([
            int(lm.x * image_width),
            int(lm.y * image_height)
        ])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = temp_landmark_list[0]
    for i in range(len(temp_landmark_list)):
        temp_landmark_list[i][0] -= base_x
        temp_landmark_list[i][1] -= base_y

    temp_landmark_list = np.array(temp_landmark_list).flatten()
    max_value = np.max(np.abs(temp_landmark_list))

    if max_value != 0:
        temp_landmark_list = temp_landmark_list / max_value

    return temp_landmark_list.tolist()

# ----------------------------
# CAMERA LOOP
# ----------------------------
cap = cv2.VideoCapture(0)
print("[INFO] Gesture-controlled VLC started")
print("[INFO] Keep VLC or browser window focused")

frame_id = 0

while True:
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    results = hand_landmarker.detect_for_video(mp_image, frame_id)
    frame_id += 1

    gesture_name = None

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            landmark_list = calc_landmark_list(image, hand_landmarks)
            pre_processed = pre_process_landmark(landmark_list)

            gesture_id = keypoint_classifier(pre_processed)
            gesture_name = keypoint_classifier_labels[gesture_id]

            # Control media
            control_media(gesture_name)

            # Draw landmarks
            for lm in landmark_list:
                cv2.circle(image, tuple(lm), 5, (0, 255, 0), -1)

    # ----------------------------
    # DISPLAY (PASTE LOCATION YOU ASKED ABOUT)
    # ----------------------------
    if gesture_name:
        cv2.putText(
            image,
            f"Gesture: {gesture_name}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Gesture VLC Controller", image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
