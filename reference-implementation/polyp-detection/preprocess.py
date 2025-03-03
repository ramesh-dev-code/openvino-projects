import os
import shutil

# Define the image dimensions
IMAGE_WIDTH = 560
IMAGE_HEIGHT = 480

def normalize_coordinates(input_folder, output_folder):
    """
    Normalize annotation coordinates to YOLO format and merge files into a single folder.

    Args:
        input_folder (str): Path to the folder containing folders of images and annotation files.
        output_folder (str): Path to the folder to save normalized annotation files and images.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all subfolders in the input folder
    for subfolder_name in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            # Loop through all .txt files in the subfolder
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.txt'):
                    input_file_path = os.path.join(subfolder_path, file_name)

                    # Generate unique file names based on subfolder and file index
                    file_index = file_name.split('.')[0]  # Extract index from file name
                    unique_base_name = f"{subfolder_name}_{file_index}"
                    output_annotation_file = os.path.join(output_folder, f"{unique_base_name}.txt")
                    output_image_file = os.path.join(output_folder, f"{unique_base_name}.jpg")

                    # Read and process each file
                    with open(input_file_path, 'r') as file:
                        lines = file.readlines()

                    # Get the class ID
                    class_id = lines[0].strip()

                    # Normalize bounding box coordinates
                    normalized_lines = []
                    for line in lines[1:]:
                        x_min, y_min, x_max, y_max = map(int, line.split())
                        x_center = ((x_min + x_max) / 2) / IMAGE_WIDTH
                        y_center = ((y_min + y_max) / 2) / IMAGE_HEIGHT
                        width = (x_max - x_min) / IMAGE_WIDTH
                        height = (y_max - y_min) / IMAGE_HEIGHT
                        normalized_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                    # Write the normalized annotations to the output file
                    with open(output_annotation_file, 'w') as output_file:
                        output_file.writelines(normalized_lines)

                    # Copy the corresponding image file to the output folder with a unique name
                    image_file_name = file_name.replace('.txt', '.jpg')
                    input_image_path = os.path.join(subfolder_path, image_file_name)
                    if os.path.exists(input_image_path):
                        shutil.copy(input_image_path, output_image_file)

# Example usage
input_folder = "Combined"
output_folder = "obj_train_data"
normalize_coordinates(input_folder, output_folder)
