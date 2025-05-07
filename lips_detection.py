# lips_detection.py - Görüntüden dudak bölgesinin tespiti

import numpy as np
import dlib
import imutils
import cv2
import os
from imutils import face_utils

def mouth_detection(shape_predictor, img, saved_name, saved_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == 'mouth':
                clone = image.copy()
                for (x, y) in shape[i:j]:
                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    roi = image[y-12:y + h + 12, x-5:x + w + 5]
                    roi = imutils.resize(roi, width=125, inter=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(saved_path, saved_name), roi)

if __name__ == "__main__":
    frames_dir = 'frames2'
    for subdir, dirs, files in os.walk(frames_dir):
        for file in files:
            if subdir != frames_dir:
                folder_name = subdir[-6:]
                saved_dir = f"lips2/{folder_name}"
                if not os.path.isdir(saved_dir):
                    os.makedirs(saved_dir)
            file_path = os.path.join(subdir, file)
            if file != '.DS_Store':
                mouth_detection('Cascade_Files/shape_predictor_68_face_landmarks.dat', file_path, file, saved_dir)