# Verilen sözlük
data = {"four": 0, "lay": 1, "t": 2, "sp": 3, "with": 4, "please": 5, "blue": 6, "zero": 7, "r": 8, "in": 9, "red": 10, "c": 11, "by": 12, "at": 13, "j": 14, "k": 15, "e": 16, "bin": 17, "m": 18, "n": 19, "i": 20, "d": 21, "y": 22, "p": 23, "now": 24, "h": 25, "eight": 26, "a": 27, "place": 28, "g": 29, "set": 30, "again": 31, "f": 32, "b": 33, "two": 34, "five": 35, "soon": 36, "seven": 37, "white": 38, "s": 39, "v": 40, "six": 41, "z": 42, "one": 43, "three": 44, "nine": 45, "q": 46, "u": 47, "l": 48, "x": 49, "o": 50, "green": 51}

bir_harf = []
iki_harf = []
uc_veya_dahaFazla = []

for key in data.keys():
    if len(key) == 1:
        bir_harf.append(key)
    elif len(key) == 2:
        iki_harf.append(key)
    else:
        uc_veya_dahaFazla.append(key)

print("1:", bir_harf, len(bir_harf))
print("2:", iki_harf, len(iki_harf))
print("3veFazlasi:", uc_veya_dahaFazla, len(uc_veya_dahaFazla))