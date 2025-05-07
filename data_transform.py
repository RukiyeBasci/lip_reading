import os
import re
import json

def get_alignment(vocab):
    alignment = {}
    path = "data\\align"

    for file in os.listdir(path):
        if file != ".DS_Store":
            file_path = path + '/' + file
            with open(file_path, mode='r') as align:
                lines = align.readlines()
                sent = []
                
                for line in lines:
                    word = re.sub('[^a-zA-Z]+', '', line)
                    if word != 'sil':
                        sent.append(vocab[word])

            alignment[file[:6]] = sent

    return alignment

if __name__ == "__main__":
    with open('vocab.txt', 'r') as f:
        vocab = json.load(f)

    a = get_alignment(vocab)
    with open('alignment.txt', 'w') as outfile:
        json.dump(a, outfile)