import os
import cv2
import torch
import numpy as np
import imutils
import dlib
from imutils import face_utils
from torchvision import transforms
from model import CNNLSTMModel
from PIL import Image
import json
from data_loader import LipreadingDataset  

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def toFrames(video_path, saved_path):
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if image is not None:
            cv2.imwrite(f'{saved_path}/frame_{count}.jpg', image)
        count += 1
    vidcap.release()

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

def extract_frames_and_lips(video_path, shape_predictor_path, temp_frames_dir="temp_frames", temp_lips_dir="temp_lips"):
    if not os.path.exists(temp_frames_dir):
        os.makedirs(temp_frames_dir)
    if not os.path.exists(temp_lips_dir):
        os.makedirs(temp_lips_dir)

    toFrames(video_path, temp_frames_dir)

    for frame_file in sorted(os.listdir(temp_frames_dir)):
        frame_path = os.path.join(temp_frames_dir, frame_file)
        mouth_detection(shape_predictor_path, frame_path, frame_file, temp_lips_dir)

def load_model(model_path, num_classes):
    model = CNNLSTMModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, frames, device, max_frames):
    while len(frames) < max_frames:
        frames.append(frames[-1])  

    frames = frames[:max_frames] 

    lengths = [len(frames)]
    frames = [test_transform(Image.open(frame).convert('RGB')) for frame in frames]
    frames = torch.stack(frames, dim=0).unsqueeze(0) 
    lengths = torch.tensor(lengths)

    frames = frames.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        outputs = model(frames, lengths)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

def split_frames_by_time_intervals(lips_dir, intervals):
    interval_frames = []
    frame_files = sorted(os.listdir(lips_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    for start, end, label in intervals:
        start_frame = int(start / 1000)
        end_frame = int(end / 1000)
        frames = frame_files[start_frame:end_frame]
        frames = [os.path.join(lips_dir, frame) for frame in frames]
        interval_frames.append((frames, label))

    return interval_frames

if __name__ == "__main__":
    video_path = "brbe9s.mpg"  
    align_path = "brbe9s.align"  
    shape_predictor_path = "Cascade_Files/shape_predictor_68_face_landmarks.dat"
    model_path = "best_model.pth"
    num_classes = 27

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, num_classes).to(device)

    extract_frames_and_lips(video_path, shape_predictor_path)

    intervals = []
    with open(align_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            start, end, label = line.strip().split()
            start = int(start)
            end = int(end)
            intervals.append((start, end, label))

    interval_frames = split_frames_by_time_intervals("temp_lips", intervals)

    vocab_path = "statistics/vocab.txt"
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}

    temp_dataset = LipreadingDataset(root_dir='extracted_frames', vocab_path='statistics\\vocab.txt', transform=test_transform)
    max_frames = temp_dataset.max_frames

    for frames, label in interval_frames:
        if label not in vocab:
            continue

        if None in frames:
            print(f"Skipping interval {label} due to missing frames.")
            continue
        
        prediction = predict(model, frames, device, max_frames)
        predicted_word = inv_vocab[prediction]
        print(f"True label: {label}, Predicted label: {predicted_word}")