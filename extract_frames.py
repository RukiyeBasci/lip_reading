import json
import os
import shutil

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_frame(src, dest):
    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))
    shutil.copy(src, dest)

def process_video(label_file, lips_dir, output_dir, vocab):
    labels = load_json(label_file)
    video_id = os.path.splitext(os.path.basename(label_file))[0]
    lips_video_dir = os.path.join(lips_dir, video_id)

    for frame_num, words in labels.items():
        for word in words:
            if len(word) > 1 and word in vocab:
                frame_path = os.path.join(lips_video_dir, f"frame_{frame_num}.jpg")
                if os.path.exists(frame_path):
                    word_output_dir = os.path.join(output_dir, word, video_id)
                    dest_path = os.path.join(word_output_dir, f"frame_{frame_num}.jpg")
                    save_frame(frame_path, dest_path)

if __name__ == "__main__":
    vocab_file = 'vocab.txt'
    labels_dir = 'data\\aligned'
    lips_dir = 'data\\lips'
    output_dir = 'extracted_frames'

    vocab = load_json(vocab_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.json'):
            process_video(os.path.join(labels_dir, label_file), lips_dir, output_dir, vocab)

    print("Frames have been extracted and organized.")