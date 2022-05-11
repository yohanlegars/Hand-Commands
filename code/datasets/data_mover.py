"""
This script is used to move annotated images from the 'not_yet_annotated' folder to the 'annotated' one. Their
corresponding .xml annotations are also moved to the 'annotated' folder.
"""
import os
import paths

if __name__ == '__main__':
    DATA_PATH = paths.DATA_PATH
    SOURCE_PATH = os.path.join(DATA_PATH, "not_yet_annotated")
    TARGET_PATH = os.path.join(DATA_PATH, "annotated")

