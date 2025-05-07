import json
import os

def load_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    return vocab

def convert_labels_to_indices(label_file, vocab):
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    updated_labels = {k: [vocab.get(word, -1) for word in words] for k, words in labels.items()}
    
    return updated_labels

def save_labels(labels, label_file):
    with open(label_file, 'w') as f:
        json.dump(labels, f, indent=4)

if __name__ == "__main__":
    vocab_file = 'vocab.txt'
    aligned_dir = 'data\\aligned'
    label_dir = 'data\\labels'

    vocab = load_vocab(vocab_file)

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    for aligned_file in os.listdir(aligned_dir):
        if aligned_file.endswith('.json'):
            video_id = os.path.splitext(aligned_file)[0]
            aligned_file_path = os.path.join(aligned_dir, aligned_file)
            labels = convert_labels_to_indices(aligned_file_path, vocab)
            save_labels(labels, os.path.join(label_dir, f"{video_id}.json"))

    print("bitti")