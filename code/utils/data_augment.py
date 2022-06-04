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

# modified from fast.ai
# partly inspired by: https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc

def create_mask(bb, x):
    """Creates a mask for the bounding box of same shape as image"""
    rows,cols,*_ = x.shape
    Y = np.zeros((rows, cols))
    bb = bb.astype(np.int)
    Y[bb[0]:bb[2], bb[1]:bb[3]] = 1.
    return Y

def mask_to_bb(Y):
    """Convert mask Y to a bounding box, assumes 0 as background nonzero object"""
    cols, rows = np.nonzero(Y)
    if len(cols)==0:
        return np.zeros(4, dtype=np.float32)
    top_row = np.min(rows)
    left_col = np.min(cols)
    bottom_row = np.max(rows)
    right_col = np.max(cols)
    return np.array([left_col, top_row, right_col, bottom_row], dtype=np.float32)

def create_bb_array(x):
    """Generates bounding box array from a train_df row"""
    return np.array([x[5],x[4],x[7],x[6]])


# def crop(im, r, c, target_r, target_c):
#     return im[r:r+target_r, c:c+target_c]

# random crop to the original size
# def random_crop(x, r_pix=8):
#     """ Returns a random crop"""
#     r, c,*_ = x.shape
#     c_pix = round(r_pix*c/r)
#     rand_r = random.uniform(0, 1)
#     rand_c = random.uniform(0, 1)
#     start_r = np.floor(2*rand_r*r_pix).astype(int)
#     start_c = np.floor(2*rand_c*c_pix).astype(int)
#     return crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)

def center_crop(x, r_pix=8):
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    return crop(x, r_pix, c_pix, r-2*r_pix, c-2*c_pix)

def rotate_cv(im, deg, y=False, mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
    """ Rotates an image by deg degrees"""
    r,c,*_ = im.shape
    M = cv2.getRotationMatrix2D((c/2,r/2),deg,1)
    if y:
        return cv2.warpAffine(im, M,(c,r), borderMode=cv2.BORDER_CONSTANT)
    return cv2.warpAffine(im,M,(c,r), borderMode=mode, flags=cv2.WARP_FILL_OUTLIERS+interpolation)

def random_cropXY(x, Y, r_pix=8):
    """ Returns a random crop"""
    r, c,*_ = x.shape
    c_pix = round(r_pix*c/r)
    rand_r = random.uniform(0, 1)
    rand_c = random.uniform(0, 1)
    start_r = np.floor(2*rand_r*r_pix).astype(int)
    start_c = np.floor(2*rand_c*c_pix).astype(int)
    xx = crop(x, start_r, start_c, r-2*r_pix, c-2*c_pix)
    YY = crop(Y, start_r, start_c, r-2*r_pix, c-2*c_pix)
    return xx, YY

def transformsXY(path, bb, transforms):
    x = cv2.imread(str(path)).astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)/255
    Y = create_mask(bb, x)
    if transforms:
        rdeg = (np.random.random()-.50)*20
        x = rotate_cv(x, rdeg)
        Y = rotate_cv(Y, rdeg, y=True)
        if np.random.random() > 0.5:
            x = np.fliplr(x).copy()
            Y = np.fliplr(Y).copy()
        x, Y = random_cropXY(x, Y)
    else:
        x, Y = center_crop(x), center_crop(Y)
    return x, mask_to_bb(Y)

def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], color=color,
                         fill=False, lw=3)

def show_corner_bb(im, bb):
    plt.imshow(im)
    plt.gca().add_patch(create_corner_rect(bb))


########################################################################################################################
# WIP ZONE: YOHAN

class RandomHorizontalFlip(torch.nn.Module):
    """
    Horizontally flip the given image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default  value is 0.5
    """
    def __init__(self, p=1):
        super().__init__()
        self.p = p

    def forward(self, dataset, num):
        image, coord_tensor, label_tensor, _ = dataset.__getitem__(num)
        transform = T.RandomHorizontalFlip(self.p)
        image_aug = transform(image)
        if image_aug != image:
            coord_tensor[0] = image.shape[2] - coord_tensor[0]
            if label_tensor == torch.Tensor([0., 0., 1., 0., 0.]):
                label_tensor = torch.Tensor([0., 0, 0., 1., 0.])

            elif label_tensor == torch.Tensor([0., 0., 0., 1., 0.]):
                label_tensor = torch.Tensor([0., 0., 1., 0., 0.])

            else:
                label_tensor = label_tensor

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


########################################################################################################################


if __name__ == '__main__':
    DATA_PATH = os.path.join(paths.DATA_PATH, "annotated")

    dataset = data_generator.HandCommandsDataset(dataset_path=DATA_PATH)
    crop_resolution = (300, 400)

    image, coord_tensor, label_tensor, name = dataset.__getitem__(4)
    print(f"{image=}")
    print(f"{image.shape=}")
    print(f"{coord_tensor=}")
    print(f"{coord_tensor.shape=}")
    print(f"{dataset.get_label_list()=}")
    print(f"{label_tensor=}")
    print(f"{label_tensor.shape=}")
    print(f"{name=}")
    print(f"{crop_resolution=}")
    print("")

    orig_img = visualization.visualize_single_instance(image, coord_tensor, label_tensor, name, dataset.label_list)
    plt.figure(name)
    plt.imshow(orig_img.permute(1, 2, 0))

    image, coord_tensor, label_tensor, name = random_crop(image, coord_tensor, label_tensor, name, crop_resolution)

    crop_img = visualization.visualize_single_instance(image, coord_tensor, label_tensor, name, dataset.label_list)
    plt.figure(name)
    plt.imshow(crop_img.permute(1, 2, 0))
    plt.show()

