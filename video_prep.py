import os
import cv2
import numpy as np
import mediapipe as mp
import glob
from tqdm import tqdm
from config import Config

class DataPreparation:
    def __init__(self):
        # MediaPipe Face Mesh yükle
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Dudak landmark indeksleri (MediaPipe'a özgü)
        self.LIPS_INDEXES = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 78, 191, 80, 81, 82
        ]
        
        # Çıktı dizinlerini oluştur
        os.makedirs(Config.OUTPUT_VIDEO_PATH, exist_ok=True)
        self.vectors_path = os.path.join(Config.DATA_PATH, "vectors")
        os.makedirs(self.vectors_path, exist_ok=True)
        
    def extract_mouth_region(self, frame):
        """Video karesinden dudak bölgesini çıkarır"""
        h, w = frame.shape[:2]
        
        # BGR'den RGB'ye çevir (MediaPipe için)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Dudak noktalarının koordinatlarını topla
        lip_points = []
        for idx in self.LIPS_INDEXES:
            lm = landmarks[idx]
            lip_points.append([int(lm.x * w), int(lm.y * h)])
            
        lip_points = np.array(lip_points)
        
        # Dudak bölgesinin sınırlarını belirle
        x_min = np.min(lip_points[:, 0]) - Config.MOUTH_WIDTH_MARGIN
        y_min = np.min(lip_points[:, 1]) - Config.MOUTH_HEIGHT_MARGIN
        x_max = np.max(lip_points[:, 0]) + Config.MOUTH_WIDTH_MARGIN
        y_max = np.max(lip_points[:, 1]) + Config.MOUTH_HEIGHT_MARGIN
        
        # Görüntü sınırlarını aşmadığından emin ol
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        # Dudak bölgesini kes
        mouth_region = frame[y_min:y_max, x_min:x_max]
        
        # Eğer geçerli bir bölge bulunamazsa None döndür
        if mouth_region.size == 0:
            return None
            
        # Belirli bir boyuta yeniden boyutlandır
        return cv2.resize(mouth_region, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
    
    def process_video(self, video_path, output_dir, vectors_dir):
        """Video işleme ve dudak bölgelerini kaydetme"""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        frame_count = 0
        
        # Video adını al
        video_name = os.path.basename(video_path).split('.')[0]
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            mouth_region = self.extract_mouth_region(frame)
            
            if mouth_region is not None:
                # Frameleri sakla
                frames.append(mouth_region)
                
                # Görüntüleri kaydet
                frame_out_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count:03d}.jpg")
                cv2.imwrite(frame_out_path, mouth_region)
                
            frame_count += 1
            
        cap.release()
        
        # Sabit uzunlukta sekans oluştur
        if len(frames) > 0:
            # Numpy array'e dönüştür
            frames_array = np.array(frames)
            
            # Sequence dosyasını kaydet
            sequence_path = os.path.join(vectors_dir, f"{video_name}_sequence.npz")
            np.savez_compressed(sequence_path, frames=frames_array)
            
            return True
        
        return False
    
    def process_speaker_videos(self, speaker_id):
        """Belirli bir konuşmacının videolarını işle"""
        speaker_dir = os.path.join(Config.VIDEO_PATH, str(speaker_id))
        video_paths = glob.glob(os.path.join(speaker_dir, '*.mpg'))
        
        successful = 0
        failed = 0
        
        for video_path in tqdm(video_paths, desc=f"Konuşmacı {speaker_id} videoları işleniyor"):
            # Çıktı dizinlerini oluştur
            output_dir = os.path.join(Config.OUTPUT_VIDEO_PATH, str(speaker_id))
            vectors_dir = os.path.join(self.vectors_path, str(speaker_id))
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(vectors_dir, exist_ok=True)
            
            # Videoyu işle
            try:
                result = self.process_video(video_path, output_dir, vectors_dir)
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Hata: {video_path} işlenirken sorun: {str(e)}")
                failed += 1
        
        print(f"Konuşmacı {speaker_id} için işleme tamamlandı. Başarılı: {successful}, Başarısız: {failed}")
    
if __name__ == "__main__":
    data_prep = DataPreparation()
    # İşlenecek konuşmacıların ID'leri
    speaker_ids = [1, 2]  # Beş konuşmacı için örnek ID listesi
    
    # Her konuşmacı için işlemi sırayla başlat
    for speaker_id in speaker_ids:
        print(f"Konuşmacı {speaker_id} için işleme başlıyor...")
        data_prep.process_speaker_videos(speaker_id)
        print(f"Konuşmacı {speaker_id} için işleme tamamlandı.")
