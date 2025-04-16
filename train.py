# train.py - Model eğitimi
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from config import Config
from models import CNNLSTMModel, TransformerModel
from dataset import get_dataloaders

def train_model(model_type='cnnlstm'):
    """Model eğitim fonksiyonu"""
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Eğitim cihazı: {device}")
    
    # Veri yükleyicileri
    train_loader, val_loader, _, word_to_index, index_to_word = get_dataloaders()
    
    # Model seçimi
    if model_type == 'cnnlstm':
        model = CNNLSTMModel(num_classes=len(word_to_index))
    else:
        model = TransformerModel(num_classes=len(word_to_index))
    
    model = model.to(device)
    
    # Kayıp fonksiyonu ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    # Eğitim için dizin oluştur
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # En iyi model için takip
    best_val_loss = float('inf')
    
    # Eğitim döngüsü
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        # Eğitim aşaması
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Gradyanları sıfırla
            optimizer.zero_grad()
            
            # Forward geçişi
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward ve optimize et
            loss.backward()
            optimizer.step()
            
            # İstatistikleri güncelle
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validasyon aşaması
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward geçişi
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # İstatistikleri güncelle
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Öğrenme oranı ayarlama
        scheduler.step(val_loss)
        
        # Geçen süreyi hesapla
        epoch_time = time.time() - start_time
        
        # Sonuçları yazdır
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - Time: {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(Config.MODEL_SAVE_PATH, f"best_{model_type}_model.pth")
            
            # Modeli kaydet
            model_state = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'word_to_index': word_to_index,
                'index_to_word': index_to_word
            }
            torch.save(model_state, model_path)
            print(f"En iyi model kaydedildi: {model_path}")
    
    print("Eğitim tamamlandı!")
    return model, word_to_index, index_to_word

def test_model(model, test_loader, device):
    """Test veri seti üzerinde modeli değerlendir"""
    model.eval()
    test_correct = 0
    test_total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test Değerlendirmesi"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward geçişi
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # İstatistikleri güncelle
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Konfüzyon matrisi için tahminleri ve etiketleri sakla
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")
    
    return test_acc, all_preds, all_labels

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dudak Okuma Modeli Eğitimi")
    parser.add_argument('--model', type=str, default='cnnlstm', choices=['cnnlstm', 'transformer'],
                       help='Kullanılacak model mimarisi (cnnlstm veya transformer)')
    args = parser.parse_args()
    
    train_model(model_type=args.model)
