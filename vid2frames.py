# vid2frames.py - OpenCV ile videoları karelere dönüştürme

import cv2
import os

def toFrames(vid, saved_path):
    vidcap = cv2.VideoCapture(vid)
    count = 0

    while True:
        success, image = vidcap.read()
        if not success:
            break
        if image is not None:
            cv2.imwrite(f'{saved_path}/frame_{count}.jpg', image)
        count += 1

if __name__ == "__main__":
    video_dir = "s14"
    for file in os.listdir(video_dir):
        if file.endswith('.mpg'):
            saved_dir = f"frames2/{file[:6]}"
            if not os.path.isdir(saved_dir):
                os.makedirs(saved_dir)

            video = f'{video_dir}/{file}'
            toFrames(video, saved_dir)