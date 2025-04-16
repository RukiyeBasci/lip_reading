# models.py - CNN+LSTM ve Transformer modelleri
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(CNNLSTMModel, self).__init__()
        
        # 3D CNN özellikleri
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Boyut hesaplama
        self.feature_size = 96 * 6 * 6  # 100x50 giriş için
        
        # CNN'den LSTM'e geçiş için linear katman
        self.lstm_input = nn.Linear(self.feature_size, 256)
        
        # LSTM katmanı
        self.lstm = nn.LSTM(input_size=256, 
                           hidden_size=256, 
                           num_layers=2, 
                           batch_first=True, 
                           dropout=0.3, 
                           bidirectional=True)
        
        # Çıkış katmanı
        self.fc = nn.Linear(256 * 2, num_classes)  # bidirectional olduğu için 2 kat
        
        # Dropout regularizasyonu
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 3D CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Boyut düzenlemesi: [batch, channels, time, height, width] -> [batch, time, features]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        time_steps = x.size(1)
        x = x.view(batch_size, time_steps, -1)
        
        # LSTM girişi için doğrusal dönüşüm
        x = self.lstm_input(x)
        x = self.dropout(x)
        
        # LSTM
        x, _ = self.lstm(x)
        
        # Son zaman adımının çıktısını alarak sınıflandırma
        x = x[:, -1, :]
        x = self.dropout(x)
        
        # Çıkış katmanı
        x = self.fc(x)
        
        return x


class TransformerModel(nn.Module):
    def __init__(self, num_classes=Config.NUM_CLASSES):
        super(TransformerModel, self).__init__()
        
        # 3D CNN özellikleri (aynı CNN+LSTM modelindeki gibi)
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(96)
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Boyut hesaplama
        self.feature_size = 96 * 6 * 6
        
        # CNN'den Transformer'a geçiş için linear katman
        self.transformer_input = nn.Linear(self.feature_size, 512)
        
        # Pozisyon kodlaması
        self.position_encoder = PositionalEncoding(d_model=512, max_len=Config.SEQUENCE_LENGTH)
        
        # Transformer encoder katmanı
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, 
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Çıkış katmanı
        self.fc = nn.Linear(512, num_classes)
        
        # Dropout regularizasyonu
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 3D CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Boyut düzenlemesi: [batch, channels, time, height, width] -> [batch, time, features]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        time_steps = x.size(1)
        x = x.view(batch_size, time_steps, -1)
        
        # Transformer girişi için doğrusal dönüşüm
        x = self.transformer_input(x)
        
        # Pozisyon kodlaması
        x = self.position_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling (son katmanın çıktılarının ortalaması)
        x = torch.mean(x, dim=1)
        
        # Dropout
        x = self.dropout(x)
        
        # Çıkış katmanı
        x = self.fc(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Transformer için sinüzoidal pozisyon kodlaması"""
    def __init__(self, d_model, max_len=Config.SEQUENCE_LENGTH):
        super(PositionalEncoding, self).__init__()
        
        # Pozisyon kodlaması matrisini oluştur
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Kayıt edilen pozisyon kodlaması (gradient hesaplanmayacak)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # Girişe pozisyon kodlaması ekle
        x = x + self.pe[:, :x.size(1), :]
        return x
