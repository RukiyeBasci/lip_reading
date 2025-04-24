import numpy as np

npz_file_path = "data/vectors/1_bbaf2n_frame_000_0.npz"

try:
    data = np.load(npz_file_path)
    print("Keys in the .npz file:", data.keys())
    for key in data.keys():
        print(f"Data under key '{key}':", data[key])
except Exception as e:
    print(f"Error loading .npz file: {e}")