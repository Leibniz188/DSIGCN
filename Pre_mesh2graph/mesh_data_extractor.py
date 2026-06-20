import os
import re

"""
This code is used to separately extract nodes, elements and matrix elements from meshes exported by finite element software, 
as preprocessing steps for converting meshes into bin-format graphs.
There are differences among various finite element software, and modifications can be made based on this code.
"""

def extract_data(file_path, output_dir):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    filename = os.path.splitext(os.path.basename(file_path))[0]  # Get the filename without extension.
    file_output_dir = os.path.join(output_dir, filename)
    os.makedirs(file_output_dir, exist_ok=True)

    node_start, node_end = None, None
    element_start, element_end = None, None
    matrix_start, matrix_end = None, None
    max_elemset_line = -1

    # Traverse file content to determine the index
    for i, line in enumerate(lines):
        if '*Node' in line:
            node_start = i + 1
        elif '*Element' in line and node_start is not None and node_end is None:
            node_end = i - 1
        elif 'C3D4' in line:
            element_start = i + 1
        elif '*Elemset' in line and element_start is not None and element_end is None:
            element_end = i - 1
        elif re.match(r'ElemSet_\d+', line):
            max_elemset_line = i  # Record the maximum ElemSet_ line number
        elif '*Phase' in line and max_elemset_line != -1:
            matrix_start = max_elemset_line + 3
            matrix_end = i - 1

    # Extract nodes, elements and matrix into separate files and save them.
    if node_start is not None and node_end is not None:
        with open(os.path.join(file_output_dir, 'node.txt'), 'w', encoding='utf-8') as f:
            f.writelines(lines[node_start:node_end + 1])

    if element_start is not None and element_end is not None:
        with open(os.path.join(file_output_dir, 'element.txt'), 'w', encoding='utf-8') as f:
            f.writelines(lines[element_start:element_end + 1])

    if matrix_start is not None and matrix_end is not None:
        with open(os.path.join(file_output_dir, 'matrix.txt'), 'w', encoding='utf-8') as f:
            f.writelines(lines[matrix_start:matrix_end + 1])

    print(f'[{filename}] processing complete.')


def process_directory(input_dir, output_dir):
    for file_name in os.listdir(input_dir):
        # Mesh file extensions vary across different finite element software, so this code only provides one parsing method.
        if file_name.endswith('_mesh.dat'):
            file_path = os.path.join(input_dir, file_name)
            if os.path.isfile(file_path):
                extract_data(file_path, output_dir)


if __name__ == "__main__":
    # Set relative paths pointing to the demo dataset in the repository
    # This allows users to run the code immediately after cloning the repo
    input_directory = '../example_data/copped_fiber/'
    output_directory = '../example_data/copped_fiber/extracted/'
    
    # Automatically create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)
    
    process_directory(input_directory, output_directory)