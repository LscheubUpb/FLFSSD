import os
import random

# Paths to the image folder and identity file
dir = "C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/data/Adience/adience_mtcnn160_png_19339"
pair_list_file = 'img.list'  # Output file

pairs = []

for file in os.scandir(dir):
    pairs.append(f"{file}\n")

# Write to pair.list
with open(pair_list_file, 'w') as f:
    for line in pairs:
        f.write(line)

print(f"Generated {len(pairs)} pairs in {pair_list_file}.")
