import os
import random

# Paths to the image folder and identity file
image_folder = 'img_align_celeba'  # Update to your actual image folder path
identity_file = 'identity_CelebA.txt'  # Update to your actual identity file path
pair_list_file = 'pair.list'  # Output file

# Load the identity file to get the list of image names and their identities
identity_map = {}
with open(identity_file, 'r') as f:
    lines = f.readlines()
    
# Create a dictionary to map identities to image names
for line in lines[1:]:  # Skip the header line
    img_name, identity = line.strip().split()
    if identity not in identity_map:
        identity_map[identity] = []
    identity_map[identity].append(img_name)

# Create pairs
pairs = []
same_person_pairs = []
different_person_pairs = []

# Create same person pairs (label = 1)
for identity, imgs in identity_map.items():
    if len(imgs) > 1:
        # Choose two random images of the same person
        chosen_pair = random.sample(imgs, 2)
        same_person_pairs.append((chosen_pair[0], chosen_pair[1], 1))

# Create different person pairs (label = 0)
image_names = list(identity_map.values())
for i in range(len(image_names)):
    img1 = random.choice(image_names)
    img2 = random.choice(image_names)
    while img1 == img2:  # Ensure different individuals
        img2 = random.choice(image_names)
    pairs.append((img1[0], img2[0], 0))  # img1[0] and img2[0] are image names

# Combine pairs
pairs += same_person_pairs
random.shuffle(pairs)

# Write to pair.list
with open(pair_list_file, 'w') as f:
    for img1, img2, label in pairs:
        f.write(f"{img1} {img2} {label}\n")

print(f"Generated {len(pairs)} pairs in {pair_list_file}.")
