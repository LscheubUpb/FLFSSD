import os
from natsort import natsorted

def get_files_in_first_x_subfolders(parent_folder, sizeInClasses):
    all_subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    selected_subfolders = all_subfolders[-sizeInClasses:]
    
    # Step 3: Collect files with absolute paths
    files_list = []
    for folder in selected_subfolders:
        files = [os.path.join(folder, file.name) for file in os.scandir(folder) if file.is_file()]
        files_list.extend(natsorted(files))
    
    return files_list

def make_imagelist(sizeInClasses):
    pair_list_file = f'SimilarityLists/{sizeInClasses}_retain.list'  # Output file
    base_folder = "./data/faces_emore"

    if(not os.path.exists(pair_list_file)):
        pairs = get_files_in_first_x_subfolders(base_folder, sizeInClasses)

        with open(pair_list_file, 'w') as f:
            for line in pairs:
                f.write(f"{line}\n")
