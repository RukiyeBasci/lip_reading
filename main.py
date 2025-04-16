# main.py - Ana program
import argparse
import os
from data_preparation import DataPreparation
from train import train_model
from evaluate import evaluate_model
from lipread_live import LipReader
from config import Config

def main():
    parser = argparse.ArgumentParser(description="Dudak Okuma Sistemi")
    parser.add_argument('--action', type=str, required=True, 
                       choices=['prepare_data', 'train', 'evaluate', 'demo'],
                       help='Gerçekleştirilecek işlem')
    
    parser.add_argument('--model_type', type=str, default='cnnlstm',
                       choices=['cnnlstm', 'transformer'],
                       help='Kullanılacak model mimarisi')
    
    parser.add_argument('--model_path', type=str, 
                       help='Kaydedilmiş/kaydedilecek model dosyası yolu')
    
    parser.add_argument('--video', type=str, 
                       help='İşlenecek video dosyası (demo için)')
    
    args = parser.parse_args()
    
    # Dizinleri oluştur
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    os.makedirs(Config.OUTPUT_DATA_PATH, exist_ok=True)
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    
    # İşlemi gerçekleştir
    if args.action == 'prepare_data':
        print("Veri hazırlanıyor...")
        data_prep = DataPreparation()
        data_prep.process_grid_corpus()
        
    elif args.action == 'train':
        print(f"Model eğitiliyor: {args.model_type}")
        model, word_to_index, index_to_word = train_model(model_type=args.model_type)
        
    elif args.action == 'evaluate':
        if not args.model_path:
            raise ValueError("Değerlendirme için --model_path belirtilmelidir!")
            
        print(f"Model değerlendiriliyor: {args.model_path}")
        test_acc, word_to_index, index_to_word = evaluate_model(
            args.model_path, model_type=args.model_type)
        
    elif args.action == 'demo':
        if not args.model_path:
            raise ValueError("Demo için --model_path belirtilmelidir!")
            
        print(f"Dudak okuma demosu başlatılıyor...")
        lip_reader = LipReader(args.model_path, model_type=args.model_type)
        
        if args.video:
            lip_reader.process_video(args.video)
        else:
            lip_reader.run_webcam()
    
    print("İşlem tamamlandı!")

if __name__ == "__main__":
    main()
