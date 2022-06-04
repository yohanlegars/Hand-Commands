from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys
import os
from yaml import parse
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "utils"))
sys.path.append(CONFIG_PATH)
from util import predict_transform
import code.confs.paths as paths


"""

YOLOv3 implementation.


"""


def parse_cfg(confile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural network to be built. 
    Block is represented as a dictionary in the list.

    """
    
    file = open(confile, 'r')
    lines = file.read().split('\n') #store the lines in a list
    lines = [x for x in lines if len(x)>0]   #get rid of the empty lines
    lines = [x for x in lines if x[0] != '#']  #get rid of comments
    lines = [x.rstrip().lstrip() for x in lines] #get rid of fringe whitespaces

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":   #This marks the start of a new block
            if len(block) !=0:  #if block is not empty, implies it is storing values of previous block
                blocks.append(block) #terminate previous block by appending it into list blocks
                block = {} # rre-init the block
            block["type"] = line[1: -1].rstrip()
        else:
            key, value = line.split("=") #separate key and value with equal sign
            block[key.rstrip()] = value.lstrip() # remove any space between key and equal sign, remove any space between equal sign and value
    
    blocks.append(block)

    return blocks

class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer).__init__()

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer).__init__()
        self.anchors = anchors

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3 #initialize channel size (RGB)
    output_filters = [] #we will append the number of output filters of each block to this list

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential() #class to sequentially execute a number of nn.Module objects.

        #check the type of block 
        #create a new module for the block
        #append to module list

        if (x["type"] == "convolutional"):
            #get the info about the layer
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            
            filters = int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["stride"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size -1) // 2
            else:
                pad = 0

            #Add the convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            #Add the batch Norm Layer
        
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm{0}".format(index), bn)

            #Check the activation
            #It is either Linear or a Leaky ReLU for YOLO
            if activation == "Leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)
        
        elif (x["type"] == "upsample"):
            
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "bilinear")
            module.add_module("upsample_{}".format(index), upsample)
        
        #nn.module for route and shortcut layers

        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(",")
            #Start of a route
            start = int(x["layers"][0])
            #end, if there exists one.
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            #Positive annotation
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            route = EmptyLayer()
            module.add_module("route_{}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]
            
        # shortcut corresponds to skip connection

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
        
        elif x["type"] == "yolo":
            mask = x['mask'].split(",")
            mask = [int(x) for x in mask]
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)



if __name__ == '__main__':

    the_path = os.path.join(paths.CONFS_PATH, "yolov3.cfg")
    blocks = parse_cfg(the_path)
    net_info, module_list = create_modules(blocks)
    print(net_info)

# class Darknet(nn.Module):
#     def __init__(self):
#         super(Darknet,self).__init__()
#
#         super(Darknet, self).__init__()
#         self.blocks = parse_cfg(confile)
#         self.net_info, self.module_list = create_modules(self.blocks)
#
#         def forward(self, x, CUDA):
#             modules = self.blocks[1:]
#             outputs = {}  #we cache the outputs for the route layers
#
#             write = 0
#             for i, module in enumerate(modules):
#                 module_type = (module["type"])
#
#                 if module_type == "convolutional" or module_type == "upsample":
#                     x = self.module_list[i](x)
#
#                 elif module_type == "route":
#                     layers = module["layers"]
#                     layers = [int(a) for a in layers]
#
#                     if (layers[0]) > 0:
#                         layers[0] = layers[0] - i
#
#                     if len(layers) == 1:
#                         x = outputs[i + (layers[0])]
#
#                     else:
#                         if (layers[1]) > 0:
#                             layers[1] = layers[1] - i
#
#                         map1 = outputs[i + layers[0]]
#                         map2 = outputs[i + layers[1]]
#                         x = torch.cat((map1, map2), 1)
#
#                 elif module_type == "shortcut":
#                     from_ = int(module["from"])
#                     x = outputs[i-1] + outputs[i+from_]
#
#                 elif module_type == 'yolo':
#                     anchors = self.module_list[i][0].anchors
#                     #get the input dimensions
#                     inp_dim = int(self.net_info["height"])
#
#                     #get the number of classes
#                     num_classes = int(module["classes"])
#
#                     #Transform
#                     x = x.data
#                     x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
#                     if not write:
#                         detections = x
#                         write = 1
#
#                     else:
#                         detections = torch.cat((detections,x), 1)
#
#                 outputs[i] = x
#
#             return detections
#
#
#






