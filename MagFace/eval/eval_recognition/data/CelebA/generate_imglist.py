import os
import random

# Paths to the image folder and identity file
pair_list_file = 'img.list'  # Output file

pairs = []

for i in range(202600):
    pairs.append(f"data/CelebA/img_align_celeba/{i:>06}.jpg\n")

# Write to pair.list
with open(pair_list_file, 'w') as f:
    for line in pairs:
        f.write(line)

print(f"Generated {len(pairs)} pairs in {pair_list_file}.")
