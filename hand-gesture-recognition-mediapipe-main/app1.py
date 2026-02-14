#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import copy
import argparse
import itertools
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import CvFpsCalc
from model import KeyPointClassifier


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = get_args()

    cap = cv.VideoCapture(args.device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, args.height)

    # ================= MediaPipe HandLandmarker =================
    base_options = python.BaseOptions(
        model_asset_path="hand_landmarker.task"
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    # ================= Models =================
    keypoint_classifier = KeyPointClassifier()

    # ================= Labels =================
    with open(
        'model/keypoint_classifier/keypoint_classifier_label.csv',
        encoding='utf-8-sig'
    ) as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

    # ================= Utils =================
    fps_calc = CvFpsCalc(buffer_len=10)

    mode = 0
    number = -1
    use_brect = True

    # Manual timestamp (IMPORTANT for VIDEO mode)
    frame_timestamp_ms = 0

    # ================= Main Loop =================
    while True:
        fps = fps_calc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break

        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        frame_timestamp_ms += 33  # ~30 FPS

        result = hand_landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(
                result.hand_landmarks,
                result.handedness
            ):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_landmarks = pre_process_landmark(landmark_list)

                logging_csv(number, mode, pre_landmarks)

                hand_sign_id = keypoint_classifier(pre_landmarks)

                debug_image = draw_bounding_rect(
                    use_brect, debug_image, brect
                )
                debug_image = draw_landmarks(
                    debug_image, landmark_list
                )
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness[0].category_name,
                    keypoint_classifier_labels[hand_sign_id],
                )

        debug_image = draw_info(
            debug_image, fps, mode, number
        )

        cv.imshow("Hand Gesture Recognition", debug_image)

    cap.release()
    cv.destroyAllWindows()


# ================= Helper Functions =================

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == ord('n'):
        mode = 0
    if key == ord('k'):
        mode = 1
    return number, mode


def calc_bounding_rect(image, landmarks):
    h, w = image.shape[:2]
    points = []
    for lm in landmarks:
        points.append([int(lm.x * w), int(lm.y * h)])
    x, y, bw, bh = cv.boundingRect(np.array(points))
    return [x, y, x + bw, y + bh]


def calc_landmark_list(image, landmarks):
    h, w = image.shape[:2]
    return [[int(lm.x * w), int(lm.y * h)] for lm in landmarks]


def pre_process_landmark(landmarks):
    base_x, base_y = landmarks[0]
    temp = [[x - base_x, y - base_y] for x, y in landmarks]
    temp = list(itertools.chain.from_iterable(temp))
    max_val = max(map(abs, temp))
    return [v / max_val for v in temp]


def logging_csv(number, mode, landmarks):
    if mode == 1 and 0 <= number <= 9:
        with open(
            'model/keypoint_classifier/keypoint.csv',
            'a',
            newline=""
        ) as f:
            csv.writer(f).writerow([number, *landmarks])


# ================= Drawing =================

def draw_landmarks(image, landmark_point):
    for point in landmark_point:
        cv.circle(image, tuple(point), 5, (255, 255, 255), -1)
        cv.circle(image, tuple(point), 5, (0, 0, 0), 1)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(
            image,
            (brect[0], brect[1]),
            (brect[2], brect[3]),
            (0, 0, 0),
            1
        )
    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(
        image,
        (brect[0], brect[1]),
        (brect[2], brect[1] - 22),
        (0, 0, 0),
        -1
    )

    text = f"{handedness}:{hand_sign_text}"
    cv.putText(
        image,
        text,
        (brect[0] + 5, brect[1] - 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    return image


def draw_info(image, fps, mode, number):
    cv.putText(
        image,
        f"FPS:{fps}",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )

    if mode == 1 and 0 <= number <= 9:
        cv.putText(
            image,
            f"LOGGING NUM:{number}",
            (10, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
    return image


if __name__ == "__main__":
    main()
