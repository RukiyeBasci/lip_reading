# evaluate.py - Model değerlendirme
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from config import Config
from models import CNNLSTMModel, TransformerModel
from dataset import get_dataloaders

def load_model(model_path, model_type='cnnlstm'):
    """Kaydedilmiş modeli yükle"""
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # Model state'i yükle
    checkpoint = torch.load(model_path, map_location=device)
    
    # Kelime indeks sözlüklerini al
    word_to_index = checkpoint['word_to_index']
    index_to_word = checkpoint['index_to_word']
    
    # Model mimarisini oluştur
    if model_type == 'cnnlstm':
        model = CNNLSTMModel(num_classes=len(word_to_index))
    else:
        model = TransformerModel(num_classes=len(word_to_index))
    
    # Modeli cihaza taşı
    model = model.to(device)
    
    # Kaydedilmiş parametreleri yükle
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model yüklendi: {model_path}")
    print(f"Epoch: {checkpoint['epoch']}, Validation Loss: {checkpoint['val_loss']:.4f}, Validation Accuracy: {checkpoint['val_acc']:.4f}")
    
    return model, word_to_index, index_to_word

def evaluate_model(model_path, model_type='cnnlstm'):
    """Modeli değerlendir ve sonuçları görselleştir"""
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    
    # Modeli yükle
    model, word_to_index, index_to_word = load_model(model_path, model_type)
    
    # Test veri yükleyicisini al
    _, _, test_loader, _, _ = get_dataloaders()
    
    # Test modunu etkinleştir
    model.eval()
    
    # Test istatistikleri
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
    
    # Test doğruluğu
    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Konfüzyon matrisini hesapla
    cm = confusion_matrix(all_labels, all_preds)
    
    # En çok karıştırılan sınıfları bul
    n_classes = len(word_to_index)
    errors = np.zeros((n_classes, n_classes))
    
    for i in range(n_classes):
        row_sum = np.sum(cm[i, :])
        if row_sum > 0:
            errors[i, :] = cm[i, :] / row_sum
    
    # Köşegen elemanlarını sıfırla (doğru tahminler)
    np.fill_diagonal(errors, 0)
    
    # En yüksek hata oranına sahip 10 sınıf çifti
    flat_errors = errors.flatten()
    top_indices = np.argsort(flat_errors)[-10:][::-1]
    
    for idx in top_indices:
        i, j = idx // n_classes, idx % n_classes
        if i != j and errors[i, j] > 0:
            print(f"'{index_to_word[i]}' {errors[i, j]:.4f} olasılıkla '{index_to_word[j]}' olarak yanlış tahmin edildi.")
    
    # Sınıflandırma raporu
    class_names = [index_to_word[i] for i in range(len(index_to_word))]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nSınıflandırma Raporu:")
    print(report)
    
    # Konfüzyon matrisini görselleştir (en çok karıştırılan 10 sınıf)
    plt.figure(figsize=(12, 10))
    
    # Sadece en çok karıştırılan 10 sınıfı seç
    confusion_idx = np.unique(np.concatenate([np.array([idx // n_classes, idx % n_classes]) for idx in top_indices]))
    selected_cm = cm[np.ix_(confusion_idx, confusion_idx)]
    selected_class_names = [class_names[i] for i in confusion_idx]
    
    sns.heatmap(selected_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=selected_class_names, 
                yticklabels=selected_class_names)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    plt.title(f'Konfüzyon Matrisi (En Çok Karıştırılan 10 Sınıf) - Test Acc: {test_acc:.4f}')
    
    # Sonuç görselini kaydet
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, f"{model_type}_confusion_matrix.png"))
    plt.close()
    
    return test_acc, word_to_index, index_to_word

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dudak Okuma Modeli Değerlendirmesi")
    parser.add_argument('--model_path', type=str, required=True, help='Model dosyasının yolu')
    parser.add_argument('--model_type', type=str, default='cnnlstm', choices=['cnnlstm', 'transformer'],
                       help='Kullanılan model mimarisi (cnnlstm veya transformer)')
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.model_type)
