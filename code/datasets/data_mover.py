"""
This script is used to move annotated images from the 'not_yet_annotated' folder to the 'annotated' one. Their
corresponding .xml annotations are also moved to the 'annotated' folder.
"""
import glob
import os
import shutil
import paths


def is_annotated(image_file):
    """
    Returns True or False, depending on whether the image has a corresponding .xml file in the same directory where it
    is located.
    :param image_name: the full path name of the image file. The function checks for an .xml file in the same directory
    :return: True or False
    """
    annot_file = image_file.split(sep=".")[0] + ".xml"
    return os.path.exists(annot_file)


if __name__ == '__main__':
    DATA_PATH = paths.DATA_PATH
    REGEX_SOURCE_IMAGES = os.path.join(DATA_PATH, "not_yet_annotated", "*.jpg")
    TARGET_PATH = os.path.join(DATA_PATH, "annotated")

    for image_file in glob.glob(REGEX_SOURCE_IMAGES):
        if is_annotated(image_file):
            image_name = os.path.basename(image_file)
            annot_file = image_file.split(sep=".")[0] + ".xml"
            annot_name = image_name.split(sep=".")[0] + ".xml"
            shutil.move(image_file, os.path.join(TARGET_PATH, image_name))
            shutil.move(annot_file, os.path.join(TARGET_PATH, annot_name))
            print("Moved: {}".format(image_name))
