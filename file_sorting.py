import os
import shutil
import re

source_dir = "path/to/mixed/files"
target_dir = "path/to/organized/files"
os.makedirs(target_dir, exist_ok=True)

num_volunteers = 70
num_tests_per_volunteer = 6

# Pattern to match volX_testY.dat (case-insensitive)
pattern = re.compile(r"(vol\d+)_test(\d+)\.dat$", re.IGNORECASE)

for filename in os.listdir(source_dir):
    if filename.lower().endswith(".dat"):
        match = pattern.match(filename)
        if match:
            vol_folder = match.group(1).lower()     # e.g., "vol1"
            test_num = match.group(2)               # e.g., "1", "2", ...
            vol_num = int(vol_folder[3:])
            test_num_int = int(test_num)
            # Validate volunteer and test numbers
            if 1 <= vol_num <= num_volunteers and 1 <= test_num_int <= num_tests_per_volunteer:
                # Create volunteer folder inside target folder
                vol_path = os.path.join(target_dir, vol_folder)
                os.makedirs(vol_path, exist_ok=True)

                # Create test folder inside volunteer folder
                test_folder = f"test{test_num}"
                test_path = os.path.join(vol_path, test_folder)
                os.makedirs(test_path, exist_ok=True)

                # Copy file to respective test folder
                src_path = os.path.join(source_dir, filename)
                dst_path = os.path.join(test_path, filename)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {filename} to {test_path}")
            else:
                print(f"Skipped (out of range): {filename}")
        else:
            print(f"Skipped (pattern mismatch): {filename}")
    else:
        print(f"Ignored (not .dat): {filename}")
