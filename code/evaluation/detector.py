from __future__ import division
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import sys
import os.path as osp
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "models"))
sys.path.append(CONFIG_PATH)
from models import darknet
import pickle as pkl
import pandas as pd

def arg_parse():
    """
    parse arguments to the detect module
    """

    parser = argparse.ArgumentParser(description = 'YOLO v3 Detection Module')

    parser.add_argument("--images", dest = 'images', help = "Images/Directory containing images to perform detection upon", default='imgs', type= str)
    parser.add_argument()
