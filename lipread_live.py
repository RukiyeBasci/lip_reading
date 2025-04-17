# lipread_live.py - Gerçek zamanlı dudak okuma
import os
import cv2
import numpy as np
import torch
import mediapipe as mp
from collections import deque
import time
from config import Config
from models import CNNLSTMModel, TransformerModel
from evaluate import load_model

class LipReader:
    def __init__(self, model_path, model_type='cnnlstm'):
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"Cihaz: {self.device}")
        
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
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
        
        # Model ve kelime sözlükleri
        self.model, self.word_to_index, self.index_to_word = load_model(model_path, model_type)
        self.model.eval()
        
        # Frame penceresi
        self.frame_window = deque(maxlen=Config.SEQUENCE_LENGTH)
        
        # Tahmin geçmişi (daha stabil çıktı için)
        self.prediction_history = []
        self.history_length = 5
        
        # FPS ölçümü
        self.prev_frame_time = 0
        self.curr_frame_time = 0
    
    def extract_mouth_region(self, frame):
        """Video karesinden dudak bölgesini çıkarır"""
        h, w = frame.shape[:2]
        
        # BGR'den RGB'ye çevir (MediaPipe için)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, frame
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Dudak noktalarının koordinatlarını topla
        lip_points = []
        for idx in self.LIPS_INDEXES:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            lip_points.append([x, y])
            # Dudak noktalarını orijinal karede çiz
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
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
        
        # Dudak bölgesini çerçeve içine al
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        # Dudak bölgesini kes
        mouth_region = frame[y_min:y_max, x_min:x_max]
        
        # Eğer geçerli bir bölge bulunamazsa None döndür
        if mouth_region.size == 0:
            return None, frame
            
        # Belirli bir boyuta yeniden boyutlandır
        mouth_region_resized = cv2.resize(mouth_region, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
        
        return mouth_region_resized, frame
    
    def preprocess_frame(self, frame):
        """Frame'i model için hazırla"""
        # [0, 255] -> [0, 1]
        frame = frame / 255.0
        
        # HWC -> CHW (PyTorch için)
        frame = frame.transpose(2, 0, 1)
        
        return frame
    
    def get_prediction(self):
        """Mevcut frame penceresi için tahmin yap"""
        if len(self.frame_window) < Config.SEQUENCE_LENGTH:
            return None
        
        # Frame penceresini uygun formata dönüştür
        frames = np.array(list(self.frame_window))
        
        # PyTorch tensörüne dönüştür
        frames_tensor = torch.FloatTensor(frames)
        
        # Batch boyutu ekle: [C, T, H, W] -> [1, C, T, H, W]
        frames_tensor = frames_tensor.unsqueeze(0)
        
        # Tensörü GPU'ya taşı
        frames_tensor = frames_tensor.to(self.device)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # En yüksek olasılıklı tahmini ve olasılığını al
            prob, pred_idx = torch.max(probs, 1)
            
            prediction = self.index_to_word[pred_idx.item()]
            confidence = prob.item()
            
            # Tahmin geçmişini güncelle
            self.prediction_history.append((prediction, confidence))
            if len(self.prediction_history) > self.history_length:
                self.prediction_history.pop(0)
            
            # Geçmiş tahminlerden en yüksek güvenilirliğe sahip olanı seç
            if len(self.prediction_history) > 0:
                best_pred = max(self.prediction_history, key=lambda x: x[1])
                return best_pred
            else:
                return (prediction, confidence)
    
    def run_webcam(self):
        """Webcam'den gerçek zamanlı dudak okuma yap"""
        cap = cv2.VideoCapture(0)  # Webcam, 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Kameradan görüntü alınamıyor!")
                break
            
            # FPS hesapla
            self.curr_frame_time = time.time()
            fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
            self.prev_frame_time = self.curr_frame_time
            
            # FPS'i ekrana yaz
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Görüntüyü yatay olarak çevir (ayna efekti)
            frame = cv2.flip(frame, 1)
            
            # Dudak bölgesini çıkar
            mouth_region, frame = self.extract_mouth_region(frame)
            
            if mouth_region is not None:
                # Dudak bölgesini ön işleme
                processed_frame = self.preprocess_frame(mouth_region)
                
                # Frame penceresine ekle
                self.frame_window.append(processed_frame)
                
                # Modelimizden tahmin yap
                prediction = self.get_prediction()
                
                if prediction:
                    word, confidence = prediction
                    # Tahmini ekrana yaz
                    text = f"Kelime: {word} ({confidence:.2f})"
                    cv2.putText(frame, text, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Dudak bölgesini ekranda göster
                mouth_display = cv2.resize(mouth_region, (200, 100))
                frame[50:150, frame.shape[1]-210:frame.shape[1]-10] = mouth_display
            
            # Canlı görüntüyü göster
            cv2.imshow('Dudak Okuma', frame)
            
            # Çıkış için 'q' tuşu
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_video(self, video_path):
        """Video dosyasından dudak okuma"""
        cap = cv2.VideoCapture(video_path)
        
        # Video özellikleri
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Çıkış video yazıcı
        output_path = os.path.splitext(video_path)[0] + "_lip_reading.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Dudak bölgesini çıkar
            mouth_region, frame = self.extract_mouth_region(frame)
            
            if mouth_region is not None:
                # Dudak bölgesini ön işleme
                processed_frame = self.preprocess_frame(mouth_region)
                
                # Frame penceresine ekle
                self.frame_window.append(processed_frame)
                
                # Modelimizden tahmin yap
                prediction = self.get_prediction()
                
                if prediction:
                    word, confidence = prediction
                    # Tahmini ekrana yaz
                    text = f"Kelime: {word} ({confidence:.2f})"
                    cv2.putText(frame, text, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Dudak bölgesini ekranda göster
                mouth_display = cv2.resize(mouth_region, (200, 100))
                frame[50:150, frame.shape[1]-210:frame.shape[1]-10] = mouth_display
            
            # İşlenmiş kareyi kaydet
            out.write(frame)
            
            # İlerleme göster
            if frame_count % 100 == 0:
                print(f"İşlenen kare: {frame_count}")
        
        cap.release()
        out.release()
        print(f"İşlenen video kaydedildi: {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gerçek Zamanlı Dudak Okuma")
    parser.add_argument('--model_path', type=str, required=True, help='Model dosyasının yolu')
    parser.add_argument('--model_type', type=str, default='cnnlstm', choices=['cnnlstm', 'transformer'],
                       help='Kullanılan model mimarisi (cnnlstm veya transformer)')
    parser.add_argument('--video', type=str, default=None, help='İşlenecek video dosyası (belirtilmezse webcam kullanılır)')
    args = parser.parse_args()
    
    lip_reader = LipReader(args.model_path, args.model_type)
    
    if args.video:
        lip_reader.process_video(args.video)
    else:
        lip_reader.run_webcam()
