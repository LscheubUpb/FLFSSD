import os
import random

# Path to the file containing the image paths
input_file = "img.list"
output_file = "pairs.list"

# Function to extract identity from file path (assuming it's the parent directory)
def extract_identity(file_path):
    return os.path.basename(os.path.dirname(file_path))

# Read the file and parse the lines
with open(input_file, "r") as file:
    lines = [line.strip() for line in file if line.strip()]

# Create a dictionary to group paths by identity
identity_dict = {}
for index, path in enumerate(lines):
    identity = extract_identity(path)
    if identity not in identity_dict:
        identity_dict[identity] = []
    identity_dict[identity].append(index)

# Generate pairs
pairs = []
for identity, indices in identity_dict.items():
    for idx in indices:
        # Select one same-identity pair if available
        same_identity_indices = [i for i in indices if i != idx]
        if same_identity_indices:
            same_idx = random.choice(same_identity_indices)
            pairs.append(f"{idx} {same_idx} 1")

        # Select 10 different-identity pairs by random sampling of different identities
        different_identity = random.choice([key for key in identity_dict.keys() if key != identity])
        different_samples = random.sample(identity_dict[different_identity], min(10, len(identity_dict[different_identity])))
        for diff_idx in different_samples:
            pairs.append(f"{idx} {diff_idx} 0")

# Write the pairs to the output file
with open(output_file, "w") as file:
    file.write("\n".join(pairs))

print(f"Pair list successfully written to {output_file}")