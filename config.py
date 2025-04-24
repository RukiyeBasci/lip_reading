import os

class Config:
    # Veri yolları
    DATA_PATH = "./data"
    VIDEO_PATH = os.path.join(DATA_PATH, "videos")
    ALIGN_PATH = os.path.join(DATA_PATH, "aligns")
    OUTPUT_VIDEO_PATH = os.path.join(DATA_PATH, "vid_out")
    
    # Model parametreleri
    NUM_CLASSES = 51  # GRID corpus için kelime sayısı
    SEQUENCE_LENGTH = 75  # Video frame uzunluğu
    IMAGE_WIDTH = 100
    IMAGE_HEIGHT = 50
    CHANNELS = 3
    
    # Eğitim parametreleri
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Dosya yolları
    MODEL_SAVE_PATH = "./models/saved"
    MEDIAPIPE_MODEL_PATH = "./models/face_landmarker.task"
    
    # Cihaz seçimi
    DEVICE = "cuda"  # "cuda" veya "cpu"
    
    # Dudak çıkarma marjları (piksel)
    MOUTH_WIDTH_MARGIN = 30
    MOUTH_HEIGHT_MARGIN = 20