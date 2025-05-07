import re
from collections import Counter

def count_numbers_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    numbers = re.findall(r'\d+', content)
    numbers = list(map(int, numbers))

    number_counts = Counter(numbers)
    return number_counts

if __name__ == "__main__":
    file_path = 'alignment.txt'
    number_counts = count_numbers_in_file(file_path)

    sorted_counts = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)

    for number, count in sorted_counts:
        print(f"{number}: {count}")