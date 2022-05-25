"""
This script is used to move annotated images from the 'not_yet_annotated' folder to the 'annotated' one. Their
corresponding annotation files are also moved to the 'annotated' folder.
"""
import glob
import os
import shutil
import code.confs.paths as paths


def is_annotated(image_file, extension="json"):
    """
    Returns True or False, depending on whether the image has a correspondin file in the same directory where it
    is located.

    :param image_file: the full path name of the image file. The function checks for an annotation file in the same
    directory, with the specified extension.
    :param extension: the file type of the annotation.
    :return: True or False
    """
    annot_file = image_file.split(sep=".")[0] + "." + extension
    return os.path.exists(annot_file)


def move_instance(image_file, target_path, extension="json"):
    """
    Moves the training instance image_file with its corresponding annotation to the target_path

    :param image_file: the full path name of the image file. The function checks for an annotation file in the same
    directory, with the specified extension.
    :param target_path: the path ehre the image file and the annotation file will be moved.
    :param extension: the file type of the annotation.
    :return: None
    """
    image_name = os.path.basename(image_file)
    annot_file = image_file.split(sep=".")[0] + "." + extension
    annot_name = image_name.split(sep=".")[0] + "." + extension
    shutil.move(image_file, os.path.join(target_path, image_name))
    shutil.move(annot_file, os.path.join(target_path, annot_name))
    print("Moved: {}".format(image_name))


def move_everything(source_path, target_path):
    """
    This function moves every file from within source_path to target_path

    :param source_path: the source directory wherein the files are present.
    :param target_path: the directory where the files should be moved.
    :return: None
    """
    for file in glob.glob(os.path.join(source_path, "*")):
        shutil.move(file, os.path.join(target_path, os.path.basename(file)))


if __name__ == '__main__':
    DATA_PATH = paths.DATA_PATH
    REGEX_SOURCE_IMAGES = os.path.join(DATA_PATH, "not_yet_annotated", "*.jpg")
    TARGET_PATH = os.path.join(DATA_PATH, "annotated")

    for image_file in glob.glob(REGEX_SOURCE_IMAGES):
        if is_annotated(image_file):
            move_instance(image_file, TARGET_PATH)
