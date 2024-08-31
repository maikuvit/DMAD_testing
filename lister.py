import os
import sys
from tqdm import tqdm

def list_images(path):
    image_paths = []

    for root, _, files in tqdm(os.walk(path)):
        for file in files:
            if (file.lower().endswith('.jpg')):
                image_paths.append(os.path.abspath(os.path.join(root, file)))

    return image_paths

def save_paths_to_file(image_paths, file_path):
    with open(file_path, 'w') as file:
        for image_path in image_paths:
            file.write(image_path + '\n')

# Get path and output path from command line arguments
path = sys.argv[1]  # The first argument is the path
output_path = sys.argv[2]  # The second argument is the output path

image_paths = list_images(path)
save_paths_to_file(image_paths, output_path)
print(f'Found {len(image_paths)} images')