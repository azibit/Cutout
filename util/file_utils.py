### Import Libraries

import csv, os, time, pickle, torch, shutil


def move_file(filename, destination_dir):

    """
    Move file to destination directory

    filename: The path of file to be moved
    destination_dir: Directory to move image into
    """

    os.rename(filename, destination_dir + "/" + filename.split("/")[-1])

def delete_dir_if_exists(directory):
    """
    Remove a directory if it exists

    dir - Directory to remove
    """

    if os.path.exists(directory):
        shutil.rmtree(directory)

def create_dir(directory):
    """
    Create directory. Deletes and recreate directory if already exists

    Parameter:
    string - directory - name of the directory to create if it does not already exist
    """

    delete_dir_if_exists(directory)
    os.makedirs(directory)

def copy_files(file_list, destination_dir):

    """
    Copy files to destination directory

    file_list: The list of files to be moved
    destination_dir: Directory to move image into
    """

    for file in file_list:
        shutil.copy(file, destination_dir)

def copy_to_other_dir(from_dir, to_dir):
    """
    Copy the content of the from directory into the to directory

    from_dir: Directory we are copying its content
    to_dir: Directory we are copying into
    """

    shutil.copytree(from_dir, to_dir)
