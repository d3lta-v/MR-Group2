import os
import sys
from pathlib import Path

def rename_images(directory, start_number=1, prefix="image", extension=".png"):
    """
    Rename files in a directory with sequential numbering.
    
    Args:
        directory: Path to the directory containing images
        start_number: Starting serial number (default: 1)
        prefix: Prefix for the filename (default: "image")
        extension: File extension to match (default: ".png")
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"Directory {directory} does not exist.")
        return
    
    # Get all files with the specified extension, sorted
    files = sorted([f for f in directory.iterdir() if f.suffix == extension and f.is_file()])
    
    if not files:
        print(f"No files with extension {extension} found in {directory}")
        return
    
    for index, file_path in enumerate(files, start=start_number):
        # Format the new name with zero-padded serial number
        serial = str(index).zfill(5)
        new_name = f"{prefix}_{serial}{extension}"
        new_path = directory / new_name
        
        file_path.rename(new_path)
        print(f"Renamed: {file_path.name} -> {new_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_image_names.py <directory> [start_number] [prefix]")
        sys.exit(1)
    
    directory = sys.argv[1]
    start_number = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    prefix = sys.argv[3] if len(sys.argv) > 3 else "image"
    
    rename_images(directory, start_number=start_number, prefix=prefix)