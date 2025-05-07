import cv2
import torch
from PIL import Image
from torchvision import transforms
from model import CNNLSTMModel
import json
import os
import time
import sys
from VideoProc.vid2frames import toFrames
from VideoProc.lips_detection import mouth_detection

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if getattr(sys, 'frozen', False):
    base = sys._MEIPASS
else:
    base = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(base, 'best_model3.pth')
VOCAB_PATH = os.path.join(base, 'statistics\\vocab.txt')
SHAPE_PREDICTOR_PATH = os.path.join(base, 'Cascade_Files/shape_predictor_68_face_landmarks.dat')
NUM_CLASSES = 27
MAX_FRAMES = 18

model = CNNLSTMModel(num_classes=NUM_CLASSES).to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    return vocab

vocab = load_vocab(VOCAB_PATH)
idx_to_word = {v: k for k, v in vocab.items()}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

RECORD_DURATION = 3
FPS = 25
NUM_FRAMES = RECORD_DURATION * FPS

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
OUTPUT_VIDEO = 'temp_video.mp4'
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (640, 480))

start_time = time.time()
frame_count = 0

while(cap.isOpened() and frame_count < NUM_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow('Kaydediliyor...', frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Kayit suresi: {elapsed_time:.2f} saniye")

cap.release()
out.release()
cv2.destroyAllWindows()

FRAMES_DIR = "temp_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)
toFrames(OUTPUT_VIDEO, FRAMES_DIR)

LIPS_DIR = "temp_lips"
os.makedirs(LIPS_DIR, exist_ok=True)

frame_files = sorted([os.path.join(FRAMES_DIR, f) for f in os.listdir(FRAMES_DIR) if f.endswith('.jpg')])
for frame_file in frame_files:
    frame_name = os.path.basename(frame_file)
    mouth_detection(SHAPE_PREDICTOR_PATH, frame_file, frame_name, LIPS_DIR)

lips_frame_list = []
lips_frame_files = sorted([os.path.join(LIPS_DIR, f) for f in os.listdir(LIPS_DIR) if f.endswith('.jpg')])

for lip_frame_file in lips_frame_files:
    try:
        lip_img = Image.open(lip_frame_file).convert('RGB')
        lip_tensor = transform(lip_img)
        lips_frame_list.append(lip_tensor)
    except FileNotFoundError:
        print(f"Hata: {lip_frame_file} bulunamadi. Bu frame atlanacak.")
        continue

while len(lips_frame_list) < MAX_FRAMES:
    lips_frame_list.append(lips_frame_list[-1])

lips_frame_list = lips_frame_list[:MAX_FRAMES]

input_tensor = torch.stack(lips_frame_list).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    lengths = torch.tensor([input_tensor.size(1)]).to(device)
    output = model(input_tensor, lengths)
    _, predicted_idx = torch.max(output, 1)
    predicted_word = idx_to_word[predicted_idx.item()]

print("Tahmin Edilen Kelime:", predicted_word)

import shutil
os.remove(OUTPUT_VIDEO)
shutil.rmtree(FRAMES_DIR)
shutil.rmtree(LIPS_DIR)