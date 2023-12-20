import os
import shutil
import sys

project_dir = os.environ.get("PROJECT_DIR")
sys.path.append(project_dir)

from program_utils import config


def add_prefix_to_videos(folder_path, prefix):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a video file (you can add more video file extensions if needed)
        if filename.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            old_path = os.path.join(folder_path, filename)

            # Create a new filename with the added suffix
            new_filename = f"{prefix}_{os.path.splitext(filename)[0]}{os.path.splitext(filename)[1]}"
            new_path = os.path.join(folder_path, new_filename)

            # Rename the file
            shutil.move(old_path, new_path)
            print(f"Renamed: {filename} to {new_filename}")


# Example: Add "D11" suffix to all video files in the specified folder
print(project_dir)
print(config.get("path").get("videos"))

folder_path = os.path.join(project_dir,
                           config.get("path").get("videos"), 'D11_success')

prefix = "D11_success"
add_prefix_to_videos(folder_path, prefix)
