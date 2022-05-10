import os

CONFIG_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.abspath(os.path.join(CONFIG_PATH, os.pardir))
DATA_PATH = os.path.abspath(os.path.join(ROOT_PATH, "data"))
CODE_PATH = os.path.abspath(os.path.join(ROOT_PATH, "code"))


if __name__ == '__main__':
    print(f"CONFIG_PATH = {CONFIG_PATH}")
    print(f"ROOT_PATH = {ROOT_PATH}")
    print(f"DATA_PATH = {DATA_PATH}")
    print(f"CODE_PATH = {CODE_PATH}")
