import os
from collections import defaultdict

def analyze_video_frame_counts(root_dir):
    video_frame_counts_per_word = {}
    words = os.listdir(root_dir)

    for word in words:
        word_dir = os.path.join(root_dir, word)
        videos = os.listdir(word_dir)
        frame_counts = defaultdict(int)

        for video in videos:
            video_dir = os.path.join(word_dir, video)
            frames = [f for f in os.listdir(video_dir) if f.endswith('.jpg') or f.endswith('.png')]
            frame_count = len(frames)
            frame_counts[frame_count] += 1

        video_frame_counts_per_word[word] = frame_counts

    return video_frame_counts_per_word

def print_video_frame_counts(video_frame_counts_per_word):
    for word, counts in video_frame_counts_per_word.items():
        print(f"Word: {word}")

        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        for frame_count, video_count in sorted_counts:
            print(f"  {frame_count}: {video_count} videos")
        print()


if __name__ == "__main__":
    root_dir = 'extracted_frames' 
    video_frame_counts_per_word = analyze_video_frame_counts(root_dir)
    print_video_frame_counts(video_frame_counts_per_word)