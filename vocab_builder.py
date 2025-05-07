import os
import re
import json

def get_vocab():
    path = "C:\\Users\\stj.rbasci\\Desktop\\lip_reading\\data\\align"
    vocab = []

    for file in os.listdir(path):
        if file != ".DS_Store":
            file_path = path + '/' + file
            with open(file_path, mode='r') as align:
                lines = align.readlines()
                sent = []
                for line in lines:
                    word = re.sub('[^a-zA-Z]+', '', line)
                    if word != 'sil':
                        sent.append(word)
                vocab += sent

    vocabulary = set(vocab)

    dictionary = {}
    count = 0
    for word in vocabulary:
        dictionary[word] = count
        count += 1
    
    return dictionary

if __name__ == "__main__":
    v = get_vocab()
    with open('vocab.txt', 'w') as outfile:
        json.dump(v, outfile)