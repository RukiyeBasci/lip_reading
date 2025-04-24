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
    
    # Konuşmacı için vektör klasörü oluşturun
    speaker_vectors_dir = os.path.join(vectors_dir, str(speaker_id))
    os.makedirs(speaker_vectors_dir, exist_ok=True)

    # Klasördeki tüm dosyaları işleyin (videolar ve frame'ler)
    for filename in os.listdir(speaker_vid_dir):
        filepath = os.path.join(speaker_vid_dir, filename)

        # Dosya bir frame ise
        if filename.endswith(".jpg"):
            frame = cv2.imread(filepath)  # Frame'i okuyun

            # Frame'i VGG16 için ön işleyin
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (224, 224))
            frame_preprocessed = preprocess_input(frame_resized)

            # VGG16 ile özellikler çıkarın
            features = model.predict(np.expand_dims(frame_preprocessed, axis=0))

            # Özellikleri .npz dosyası olarak kaydedin
            npz_filename = f"{filename[:-4]}.npz"  # .jpg uzantısını çıkarın
            npz_filepath = os.path.join(speaker_vectors_dir, npz_filename)
            np.savez_compressed(npz_filepath, X=features)

        # Dosya bir video ise (bu kısım artık kullanılmayacak)
        else:
            pass  # Video işleme kısmını atlayın

print("Tüm videolar işlendi ve vektörler kaydedildi.")