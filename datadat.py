import os
import shutil

source_parent_dir = "path/to/source/parent"  # Folder containing V3, V4, ... V61 folders
target_parent_dir = "path/to/target/parent"  # Folder where new V3, V4, ... folders with .dat files will be created
os.makedirs(target_parent_dir, exist_ok=True)

vol_start = 3
vol_end = 61

for vol_num in range(vol_start, vol_end + 1):
    source_folder = os.path.join(source_parent_dir, f"V{vol_num}")
    target_folder = os.path.join(target_parent_dir, f"V{vol_num}")
    os.makedirs(target_folder, exist_ok=True)

    if os.path.isdir(source_folder):
        for filename in os.listdir(source_folder):
            if filename.lower().endswith(".dat"):
                src_path = os.path.join(source_folder, filename)
                dst_path = os.path.join(target_folder, filename)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {filename} from {source_folder} to {target_folder}")
    else:
        print(f"Source folder not found: {source_folder}")
