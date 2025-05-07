import os
import json

def time_to_frames(time, fps):
    return int(time * fps / 1000)

def create_aligned_file(time_intervals, words, fps, output_path):
    aligned_data = {}
    
    for i, interval in enumerate(time_intervals):
        start_time, end_time = interval
        start_frame = time_to_frames(start_time, fps)
        end_frame = time_to_frames(end_time, fps)
        
        for frame in range(start_frame, end_frame):
            aligned_data[frame] = [words[i]]
    
    with open(output_path, 'w') as f:
        json.dump(aligned_data, f, indent=4)

if __name__ == "__main__":
    aligns_path = "data\\align"
    aligned_output_path = "data\\aligned"
    fps = 1.01  # Hesaplanan FPS değeri

    if not os.path.exists(aligned_output_path):
        os.makedirs(aligned_output_path)

    for file in os.listdir(aligns_path):
        if file.endswith(".align"):
            file_path = os.path.join(aligns_path, file)
            with open(file_path, 'r') as align_file:
                lines = align_file.readlines()
                time_intervals = []
                words = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        start_time = int(parts[0])
                        end_time = int(parts[1])
                        word = parts[2]
                        time_intervals.append((start_time, end_time))
                        words.append(word)
            
            output_file = os.path.join(aligned_output_path, file.replace(".align", ".json"))
            create_aligned_file(time_intervals, words, fps, output_file)
