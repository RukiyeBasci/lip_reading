import os
import numpy as np
import cv2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Veri klasörlerinin yollarını tanımlayın
data_dir = "data"
vid_out_dir = os.path.join(data_dir, "vid_out")
vectors_dir = os.path.join(data_dir, "vectors")

# Vektör klasörünü oluşturun (eğer yoksa)
os.makedirs(vectors_dir, exist_ok=True)

# Konuşmacı ID'lerini belirleyin
speaker_ids = list(range(1, 44))
speaker_ids.remove(21)

# VGG16 modelini yükleyin
model = VGG16(weights='imagenet', include_top=False, pooling='avg')

# Her konuşmacı için videoları işleyin
for speaker_id in speaker_ids:
    # Konuşmacıya ait videoların bulunduğu klasörü belirleyin
    speaker_vid_dir = os.path.join(vid_out_dir, str(speaker_id))

    # Klasördeki tüm videoları işleyin
    for video_filename in os.listdir(speaker_vid_dir):
        # Video dosya yolunu belirleyin
        video_filepath = os.path.join(speaker_vid_dir, video_filename)

        # Video dosyasını açın
        cap = cv2.VideoCapture(video_filepath)

        # Her frame için bir vektör oluşturun ve kaydedin
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame'i VGG16 için ön işleyin
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frame_preprocessed = preprocess_input(frame_resized)

            # VGG16 ile özellikler çıkarın
            features = model.predict(np.expand_dims(frame_preprocessed, axis=0))

            # Özellikleri .npz dosyası olarak kaydedin
            npz_filename = f"{speaker_id}_{video_filename[:-4]}_{frame_count}.npz"
            npz_filepath = os.path.join(vectors_dir, npz_filename)
            np.savez_compressed(npz_filepath, X=features)

            frame_count += 1

        # Video dosyasını kapatın
        cap.release()

print("Tüm videolar işlendi ve vektörler kaydedildi.")