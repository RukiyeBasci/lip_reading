# dataset.py - PyTorch dataset sınıfı
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from config import Config

class LipReadingDataset(Dataset):
    def __init__(self, data_path, transform=None, split='train'):
        """
        Args:
            data_path: İşlenmiş veri dosyalarının yolu
            transform: Veri arttırma dönüşümleri
            split: 'train', 'val' veya 'test'
        """
        self.data_path = data_path
        self.transform = transform
        self.split = split
        
        # Tüm sequence dosyalarını bul
        self.sequence_files = sorted(glob.glob(os.path.join(data_path, '*/*_sequence.npy')))
        
        # Eğitim/Validasyon/Test bölümlerine ayır
        if split == 'train':
            self.sequence_files = self.sequence_files[:int(len(self.sequence_files) * 0.8)]
        elif split == 'val':
            self.sequence_files = self.sequence_files[int(len(self.sequence_files) * 0.8):int(len(self.sequence_files) * 0.9)]
        else:  # test
            self.sequence_files = self.sequence_files[int(len(self.sequence_files) * 0.9):]
            
        # Etiketleri ve kelime indekslerini oluştur
        self.word_to_index = {}
        self.index_to_word = {}
        
        # GRID Corpus kelime listesi
        self.words = ['bin', 'blue', 'green', 'red', 'white', 'at', 'by', 'in', 'with', 'a', 'b', 'c', 
                     'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 
                     't', 'u', 'v', 'w', 'x', 'y', 'z', 'zero', 'one', 'two', 'three', 'four', 'five', 
                     'six', 'seven', 'eight', 'nine', 'again', 'now', 'please', 'soon']
        
        for i, word in enumerate(self.words):
            self.word_to_index[word] = i
            self.index_to_word[i] = word
            
    def __len__(self):
        return len(self.sequence_files)

    def __getitem__(self, idx):
        sequence_path = self.sequence_files[idx]
        
        # Sequence'i yükle
        sequence = np.load(sequence_path)
        
        # Etiket dosya yolunu belirle
        label_path = sequence_path.replace('_sequence.npy', '_label.txt')
        
        # Etiketi yükle
        with open(label_path, 'r') as f:
            word = f.read().strip()
            
        # Kelime indeksine dönüştür
        label = self.word_to_index.get(word, 0)  # Bilinmeyen kelimeleri 0 indeksine ata
        
        # Normalizasyon: [0, 255] -> [0, 1]
        sequence = sequence / 255.0
        
        # Veri artırma uygula
        if self.transform:
            frames = []
            for i in range(sequence.shape[0]):
                frame = self.transform(sequence[i])
                frames.append(frame)
            sequence = np.array(frames)
            
        # PyTorch tensörüne dönüştür
        sequence_tensor = torch.FloatTensor(sequence)
        
        # Kanalları ilk boyuta taşı: [T, H, W, C] -> [C, T, H, W]
        sequence_tensor = sequence_tensor.permute(3, 0, 1, 2)
        
        return sequence_tensor, torch.tensor(label, dtype=torch.long)

def get_dataloaders():
    """Eğitim, validasyon ve test için DataLoader'ları oluşturur"""
    from torchvision import transforms
    
    # Veri artırma işlemleri
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset'leri oluştur
    train_dataset = LipReadingDataset(Config.OUTPUT_DATA_PATH, transform=train_transform, split='train')
    val_dataset = LipReadingDataset(Config.OUTPUT_DATA_PATH, transform=val_transform, split='val')
    test_dataset = LipReadingDataset(Config.OUTPUT_DATA_PATH, transform=val_transform, split='test')
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset.word_to_index, train_dataset.index_to_word
