# Importing the necessary libraries
from __future__ import annotations

import os
import shutil
from math import ceil
from pathlib import Path


def split_and_symlink_folders(main_folder_path):
    # List all subfolders in the main folder
    subfolders = [f.name for f in os.scandir(main_folder_path) if f.is_dir()]

    # Count the total number of subfolders
    total_folders = len(subfolders)

    # Calculate the size of each split
    split_size = ceil(total_folders / 5)

    # Create 5 splits of the subfolders
    splits = [subfolders[i : i + split_size] for i in range(0, len(subfolders), split_size)]

    # Create 5 resulting folders, each containing 4 of the 5 splits
    for i in range(5):
        parent_folder = Path(main_folder_path).parent
        # Create the resulting folder
        result_folder_name = f"{parent_folder}/result_{i + 1}"

        os.makedirs(result_folder_name, exist_ok=True)

        # Determine which splits to include in the resulting folder
        splits_to_include = [splits[j] for j in range(5) if j != i]

        # Flatten the list of splits to include
        folders_to_include = [folder for split in splits_to_include for folder in split]

        # Create symlinks for each of the folders to include
        for folder in folders_to_include:
            source_folder = os.path.join(main_folder_path, folder)
            destination_folder = os.path.join(result_folder_name, folder)
            os.symlink(source_folder, destination_folder, target_is_directory=True)


# Define the main folder path (Replace this with the path to your main folder)
main_folder_path = "/projects/datashare/tio/AutoPET2/FDG-PET-CT-Lesions/"

# Run the function to perform the split and symlink operation
split_and_symlink_folders(main_folder_path)
