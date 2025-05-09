import os
from natsort import natsorted

# Paths to the image folder and identity file
pair_list_file = 'img.list'  # Output file
base_folder = "C:/Users/leosc/Documents/_wichtigeDokumente/Bachelorarbeit/selective-synaptic-dampening-main/src/data/105_classes_pins_dataset"

def get_files_in_first_x_subfolders(parent_folder, x):
    # Step 1: Get all subdirectories in the parent folder
    all_subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    
    # Step 2: Limit to the first x subdirectories
    # selected_subfolders = all_subfolders[len(all_subfolders)-x:]
    selected_subfolders = all_subfolders#[:x]
    
    # Step 3: Collect files with absolute paths
    files_list = []
    for folder in selected_subfolders:
        files = [os.path.join(folder, file.name) for file in os.scandir(folder) if file.is_file()]
        files_list.extend(natsorted(files))
    
    return files_list

x = 500
pairs = get_files_in_first_x_subfolders(base_folder, x)

# Write to pair.list
with open(pair_list_file, 'w') as f:
    for line in pairs:
        f.write(f"{line}\n")

print(f"Generated {len(pairs)} pairs in {pair_list_file}.")
