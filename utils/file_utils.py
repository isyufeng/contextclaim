import os
import json

def read_jsonl_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            yield json.loads(line)


def get_filename(file_path):
    """
    Extract the file name from a given file path.

    :param file_path: The full path to the file
    :return: The file name with extension
    """
    return os.path.basename(file_path)


def get_filename_without_extension(file_path):
    """
    Extract the file name without extension from a given file path.

    :param file_path: The full path to the file
    :return: The file name without extension
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def get_extension(file_path):
    """
    Extract the file extension from a given file path.

    :param file_path: The full path to the file
    :return: The file extension (including the dot)
    """
    return os.path.splitext(file_path)[1]


# Method 1: Simple write
def simple_write(filename, text):
    with open(filename, 'w') as file:
        file.write(text)

# Method 2: Append to file
def append_to_file(filename, text):
    with open(filename, 'a') as file:
        file.write(text)

# Method 3: Write multiple lines
def write_lines(filename, lines):
    with open(filename, 'w') as file:
        for line in lines:
            file.write(line + '\n')

# Method 4: Using print function
def write_using_print(filename, text):
    with open(filename, 'w') as file:
        print(text, file=file)

# # Example usage:
# simple_write('example1.txt', 'Hello, World!')
# append_to_file('example2.txt', 'This is a new line.\n')
# write_lines('example3.txt', ['Line 1', 'Line 2', 'Line 3'])
# write_using_print('example4.txt', 'Using print function to write.')

# Bonus: Writing with error handling
def safe_write(filename, text):
    try:
        with open(filename, 'w') as file:
            file.write(text)
    except IOError as e:
        print(f"An error occurred while writing to {filename}: {e}")
    else:
        print(f"Successfully wrote to {filename}")

# safe_write('example5.txt', 'Writing with error handling.')