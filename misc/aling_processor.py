import os
import re
import json
from tqdm import tqdm

class AlingProcessor:
    def __init__(self, alings_path, vocab_path):
        self.alings_path = alings_path
        self.vocab = self.load_vocab(vocab_path)  # Kelimeleri indekslerle eşleştirecek sözlük
        self.alignments = {}  # Kişi bazlı kelime indekslerini saklayacak

    def load_vocab(self, vocab_path):
        """Mevcut vocab.json dosyasını yükler."""
        with open(vocab_path, "r") as vocab_file:
            vocab = json.load(vocab_file)
        print(f"Sözlük yüklendi: {len(vocab)} kelime.")
        return vocab

    def build_alignments(self):
        """Align dosyalarını kişi bazlı zaman aralıkları ve kelime indeksleriyle kaydeder."""
        for speaker_id in tqdm(sorted(os.listdir(self.alings_path)), desc="Konuşmacılar işleniyor"):
            speaker_path = os.path.join(self.alings_path, speaker_id, "align")
            if not os.path.isdir(speaker_path):
                continue  # Eğer dosya değilse geç

            # Kişiye özel hizalamalar için yeni bir anahtar oluştur
            self.alignments[speaker_id] = {}

            for file_name in os.listdir(speaker_path):
                file_path = os.path.join(speaker_path, file_name)
                if not file_name.endswith(".align"):
                    continue  # Sadece .align dosyalarını işliyoruz
                
                alignment_with_timing = []
                with open(file_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            start_time, end_time, word = int(parts[0]), int(parts[1]), parts[2]
                            # 'sil' kelimesini atla
                            if word != "sil":
                                # Kelimeyi indekse dönüştür
                                word_idx = self.vocab.get(word, None)
                                if word_idx is not None:
                                    alignment_with_timing.append([start_time, end_time, word_idx])

                # Dosya adına göre hizalamayı kaydet
                video_id = os.path.splitext(file_name)[0]  # Örn: "bbaf2n.align" -> "bbaf2n"
                self.alignments[speaker_id][video_id] = alignment_with_timing

        # Hizalamaları diske kaydet
        with open("alignment.json", "w") as align_file:
            json.dump(self.alignments, align_file, indent=4)
        print("Hizalamalar 'alignment.json' dosyasına kaydedildi.")

if __name__ == "__main__":
    # Aling dosyalarının bulunduğu ana klasör
    ALINGS_PATH = "C:\\Users\\stj.rbasci\\Downloads\\lip_reading-main\\lip_reading-main\\data\\alings"
    # Sözlük dosyasının yolu
    VOCAB_PATH = "C:\\Users\\stj.rbasci\\Downloads\\lip_reading-main\\lip_reading-main\\misc\\vocab.json"

    processor = AlingProcessor(ALINGS_PATH, VOCAB_PATH)
    print("Aling dosyaları işleniyor...")
    processor.build_alignments()  # Hizalamaları oluştur
    print("Tüm işlemler tamamlandı!")