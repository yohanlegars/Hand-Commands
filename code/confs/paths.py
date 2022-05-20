import os

CONFS_PATH = os.path.abspath(os.path.dirname(__file__))
CODE_PATH = os.path.abspath(os.path.join(CONFS_PATH, os.pardir))
ROOT_PATH = os.path.abspath(os.path.join(CODE_PATH, os.pardir))
DATA_PATH = os.path.abspath(os.path.join(ROOT_PATH, "data"))

if __name__ == '__main__':
    print(f"CONFIG_PATH = {CONFS_PATH}")
    print(f"ROOT_PATH = {ROOT_PATH}")
    print(f"DATA_PATH = {DATA_PATH}")
    print(f"CODE_PATH = {CODE_PATH}")
