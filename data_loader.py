import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import json

class LipreadingDataset(Dataset):
    def __init__(self, root_dir, vocab_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        self.max_frames = self._calculate_max_frames()
        self.vocab = self._load_vocab(vocab_path)
        self._prepare_dataset()

    def _load_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        return vocab

    def _calculate_max_frames(self):
        max_len = 0
        for word in os.listdir(self.root_dir):
            word_dir = os.path.join(self.root_dir, word)
            for video in os.listdir(word_dir):
                video_dir = os.path.join(word_dir, video)
                frames = [f for f in os.listdir(video_dir) if f.endswith('.jpg') or f.endswith('.png')]
                max_len = max(max_len, len(frames))
        return max_len

    def _prepare_dataset(self):
        words = os.listdir(self.root_dir)
        for word in words:
            if word not in self.vocab:
                continue
            label = self.vocab[word]
            word_dir = os.path.join(self.root_dir, word)
            for video in os.listdir(word_dir):
                video_dir = os.path.join(word_dir, video)
                frames = sorted([os.path.join(video_dir, frame) for frame in os.listdir(video_dir) if frame.endswith('.jpg') or frame.endswith('.png')])

                while len(frames) < self.max_frames:
                    frames.append(frames[-1])  

                self.data.append(frames)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = self.data[idx]
        label = self.labels[idx]

        images = []
        for frame in frames:
            image = Image.open(frame).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)

        images = torch.stack(images, dim=0)
        return images, label, len(frames)


# Veri artırma transform'ları (eğitim için)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomApply([transforms.RandomRotation(10),
                           transforms.RandomHorizontalFlip()], p=0.7),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sadece yeniden boyutlandırma ve normalizasyon (doğrulama ve test için)
val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset oluşturma
full_dataset = LipreadingDataset(root_dir='extracted_frames', vocab_path='statistics\\vocab.txt', transform=train_transform)

# Veri setini train, validation ve test olarak ayırma
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

train_dataset.dataset.transform = train_transform  # training data için augmentation
val_dataset.dataset.transform = val_test_transform # validation data için augmentation yok
test_dataset.dataset.transform = val_test_transform # test data için augmentation yok

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return images, labels, lengths

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn) #shuffle=False validation için
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn) #shuffle=False test için