import random
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional
import code.datasets.data_generator as data_generator
import code.utils.visualization as visualization
import code.confs.paths as paths
import torch

def random_hflip(image, coord_tensor, label_tensor, name, flip_ratio=0.5):
    """
    This function performs a random horizontal flip of the image and its corresponding labels

    :param image: image tensor
    :param coord_tensor: coordinate tensor
    :param label_tensor: label tensor
    :param name: training instance name
    :return: a new training instance
    """
    flip = random.choices(population=(True, False), weights=(flip_ratio, 1-flip_ratio))[0]

    if flip:
        image_aug = torchvision.transforms.functional.hflip(image)
        coord_tensor[0] = image.shape[2] - coord_tensor[0]
        if label_tensor[2] == 1.:
            label_tensor = torch.Tensor([0., 0., 0., 1., 0.])

        elif label_tensor[3] == 1.:
            label_tensor = torch.Tensor([0., 0., 1., 0., 0.])

        name += "_flip"

    else:
        image_aug = image

    return image_aug, coord_tensor, label_tensor, name


def crop(img_tensor, coord_tensor, label_tensor, instance_name, crop_coords):
    """
    This function crops the image and modifies its label coordinates accordingly

    :param img_tensor: image tensor
    :param coord_tensor: coordinate tensor
    :param label_tensor: label tensor
    :param instance_name: training instance name
    :param crop_coords: coordinates of the cropping, int, specified as: [x left corner, y top corner, pixel width, pixel height]
    :return:
    """

    x_left, y_top, width, height = crop_coords

    img_tensor = img_tensor[:, y_top:y_top+height, x_left:x_left+width]
    coord_tensor[0] -= x_left
    coord_tensor[1] -= y_top
    coord_tensor = visualization.center_tensor_to_bbox_tensor(coord_tensor)
    coord_tensor = torch.clamp(coord_tensor, min=torch.zeros(4), max=torch.tensor([width, height, width, height]))
    coord_tensor = visualization.bbox_tensor_to_center_tensor(coord_tensor)

    return img_tensor, coord_tensor, label_tensor, instance_name


def random_crop(img_tensor, coord_tensor, label_tensor, instance_name, cropped_img_resolution):
    """
    This function performs a random crop of the image and its labels, forcing a specific output image resolution

    :param img_tensor: image tensor
    :param coord_tensor: coordinate tensor
    :param label_tensor: label tensor
    :param instance_name: training instance name
    :param cropped_img_resolution: tuple of 2 ints: [pixel height, pixel width]
    :return:
    """
    x_range = img_tensor.shape[2] - cropped_img_resolution[1]
    y_range = img_tensor.shape[1] - cropped_img_resolution[0]
    x_left = random.randrange(x_range)
    y_top = random.randrange(y_range)

    crop_coords = [x_left, y_top, cropped_img_resolution[1], cropped_img_resolution[0]]

    instance_name += "_crop"

    return crop(img_tensor, coord_tensor, label_tensor, instance_name, crop_coords)


def rotate(img_tensor, coord_tensor, label_tensor, instance_name, angle):   #TODO
    pass


def random_rotate():        #TODO
    pass


def random_augment(img_tensor, coord_tensor, label_tensor, instance_name, cropped_img_resolution):
    """
    This function performs a random data augmentation on one training instance.
    It sequentially applies a random cropping and a random flip of the image.

    :return: a new, augmented, training instance
    """
    img_tensor, coord_tensor, label_tensor, instance_name = random_crop(img_tensor, coord_tensor, label_tensor, instance_name, cropped_img_resolution)
    img_tensor, coord_tensor, label_tensor, instance_name = random_hflip(img_tensor, coord_tensor, label_tensor, instance_name)

    return img_tensor, coord_tensor, label_tensor, instance_name


if __name__ == '__main__':
    DATA_PATH = os.path.join(paths.DATA_PATH, "annotated")

    dataset = data_generator.HandCommandsDataset(dataset_path=DATA_PATH)
    crop_resolution = (430, 590)

    image, coord_tensor, label_tensor, name = dataset.__getitem__(25)
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

    image, coord_tensor, label_tensor, name = random_augment(image, coord_tensor, label_tensor, name, crop_resolution)

    mod_img = visualization.visualize_single_instance(image, coord_tensor, label_tensor, name, dataset.label_list)
    plt.figure(name)
    plt.imshow(mod_img.permute(1, 2, 0))
    plt.show()
