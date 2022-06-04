"""

TODO: RANDOM FLIP
TODO: RANDOM ANGLE (Maybe, if we have the time)

"""

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import code.datasets.data_generator as data_generator
import code.utils.visualization as visualization
import code.confs.paths as paths
import torch
import torchvision.transforms as T

########################################################################################################################
# WIP ZONE: YOHAN




def Random_Horizontal_Flip(dataset, num):
    image, coord_tensor, label_tensor, _ = dataset.__getitem__(num)
    k = random.randint(0,1)
    p = 1

    if k == 1:
        transform = T.RandomHorizontalFlip(p)
        image_aug = transform(image)
        coord_tensor[0] = image.shape[2] - coord_tensor[0]
        if label_tensor[2] == 1.:
            label_tensor = torch.Tensor([0., 0, 0., 1., 0.])

        elif label_tensor[3] == 1.:
            label_tensor = torch.Tensor([0., 0., 1., 0., 0.])
    else:
        image_aug = image

    return image_aug, image, coord_tensor, label_tensor


########################################################################################################################
# WIP ZONE: PAUL

def crop(img_tensor, coord_tensor, label_tensor, instance_name, crop_coords):

    x_left, y_top, width, height = crop_coords

    img_tensor = img_tensor[:, y_top:y_top+height, x_left:x_left+width]
    coord_tensor[0] -= x_left
    coord_tensor[1] -= y_top
    coord_tensor = visualization.center_tensor_to_bbox_tensor(coord_tensor)
    coord_tensor = torch.clamp(coord_tensor, min=torch.zeros(4), max=torch.tensor([width, height, width, height]))
    coord_tensor = visualization.bbox_tensor_to_center_tensor(coord_tensor)

    return img_tensor, coord_tensor, label_tensor, instance_name

def random_crop(img_tensor, coord_tensor, label_tensor, instance_name, cropped_img_resolution):

    # cropped_img_resolution shape [H, W]
    x_range = img_tensor.shape[2] - cropped_img_resolution[1]
    y_range = img_tensor.shape[1] - cropped_img_resolution[0]
    x_left = random.randrange(x_range)
    y_top = random.randrange(y_range)

    crop_coords = [x_left, y_top, cropped_img_resolution[1], cropped_img_resolution[0]]

    instance_name += "_crop"

    return crop(img_tensor, coord_tensor, label_tensor, instance_name, crop_coords)

def rotate(img_tensor, coord_tensor, label_tensor, instance_name, angle):
    pass

def random_rotate():
    pass

########################################################################################################################


if __name__ == '__main__':
    DATA_PATH = os.path.join(paths.DATA_PATH, "annotated")

    dataset = data_generator.HandCommandsDataset(dataset_path=DATA_PATH)
    #crop_resolution = (300, 400)


    image, coord_tensor, label_tensor, name = dataset.__getitem__(4)
    print(f"{image=}")
    print(f"{image.shape=}")
    print(f"{coord_tensor=}")
    print(f"{coord_tensor.shape=}")
    print(f"{dataset.get_label_list()=}")
    print(f"{label_tensor=}")
    print(f"{label_tensor.shape=}")
    print(f"{name=}")
   # print(f"{crop_resolution=}")
    print("")

    num = 2
    image_aug, image, coord_tensor, label_tensor = Random_Horizontal_Flip(dataset, num)
    images = [image_aug, image]
    for i in range(0,len(images)):
        plt.imshow(images[i].permute(1,2,0))
        plt.show()




    # orig_img = visualization.visualize_single_instance(image, coord_tensor, label_tensor, name, dataset.label_list)
    # plt.figure(name)
    # plt.imshow(orig_img.permute(1, 2, 0))
    #
    # image, coord_tensor, label_tensor, name = random_crop(image, coord_tensor, label_tensor, name, crop_resolution)
    #
    # crop_img = visualization.visualize_single_instance(image, coord_tensor, label_tensor, name, dataset.label_list)
    # plt.figure(name)
    # plt.imshow(crop_img.permute(1, 2, 0))
    # plt.show()

