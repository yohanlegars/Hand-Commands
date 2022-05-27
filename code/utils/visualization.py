import torch
import torchvision
import code.datasets.data_generator as data_generator
import code.confs.paths as paths
import os
import matplotlib.pyplot as plt
import configargparse


def random_bbox_tensor(B, H, W):
    """
    Create a random bounding box tensor

    :param B: batch size
    :param H: maximum pixel height
    :param W: maximum pixel width
    :return: the random bounding box tensor
    """

    x_values = torch.rand(B, 2) * W
    y_values = torch.rand(B, 2) * H

    x_values, _ = torch.sort(x_values)
    y_values, _ = torch.sort(y_values)

    x_mins, x_maxs = x_values[:, 0], x_values[:, 1]
    y_mins, y_maxs = y_values[:, 0], y_values[:, 1]

    return torch.stack((x_mins, y_mins, x_maxs, y_maxs), axis=-1)


def center_tensor_to_bbox_tensor(coord_tensor):
    """
    This function can be used to convert a coordinate tensor, into its corresponding bounding box tensor

    :param coord_tensor: tensor([x, y, width, height])
    :return: tensor([x_min, y_min, x_max, y_max])
    """
    center_x, center_y, width, height = coord_tensor.unbind(dim=-1)
    x_min = center_x - 0.5 * width
    x_max = center_x + 0.5 * width
    y_min = center_y - 0.5 * height
    y_max = center_y + 0.5 * height
    bbox_tensor = torch.stack((x_min, y_min, x_max, y_max), axis=-1)
    return bbox_tensor


def bbox_tensor_to_center_tensor(bbox_tensor):
    """
    This function can be used to convert a coordinate tensor, into its corresponding bounding box tensor

    :param bbox_tensor: tensor([x_min, y_min, x_max, y_max])
    :return: tensor([x, y, width, height])
    """
    x_min, y_min, x_max, y_max = bbox_tensor[:, 0], bbox_tensor[:, 1], bbox_tensor[:, 2], bbox_tensor[:, 3]
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    coord_tensor = torch.stack((center_x, center_y, width, height), axis=-1)
    return coord_tensor


def label_tensor_to_string(label_tensor: torch.Tensor, label_list):
    """
    This function converts a tensor containing the representation of a label or a prediction (ie, either a one-hot
    encoded tensor of a label, or a tensor containing probabilities for belonging to specific classes), into its
    corresponding, human-readable label. It does so by means of using a list of labels.

    :param label_tensor: the tensor representation of the label/prediction
    :param label_list: a list of strings, the order of the list determines the indices for class determination
    :return: the corresponding label/prediction in string format, according to its tensor representation
    """
    label_tensor = label_tensor.clone().detach()
    label_tensor = label_tensor.int()
    label_tensor = label_tensor.argmax(-1)
    label = label_list[label_tensor]
    return label


def visualize_single_instance(dataset, idx):
    """
    This function can be used to render an image with corresponding ground truth coordinates and label. The instance is
    taken from a specified dataset

    :param dataset: the dataset from which to extract an instance.
    :param idx: index of the instance, present within the provided dataset.
    :return: a tensor representation of the drawn on image. (use tensor.permute(1, 2, 0) for displaying with matplotlib)
    """
    image, coord_tensor, label_tensor, _ = dataset.__getitem__(idx)
    coord_tensor = torch.reshape(coord_tensor, (1, -1))
    bbox = center_tensor_to_bbox_tensor(coord_tensor)
    bbox = torch.reshape(bbox, (1, -1))
    label = label_tensor_to_string(label_tensor, dataset.label_list)
    label = [label]

    visual = torchvision.utils.draw_bounding_boxes(image=image, boxes=bbox, labels=label, colors="red", font_size=60)
    return visual


if __name__ == '__main__':
    path = os.path.join(paths.DATA_PATH, "annotated")

    parser = configargparse.ArgumentParser(default_config_files=[os.path.join(paths.CONFS_PATH, "training.conf")])
    parser.add_argument('--LABELS', type=str, nargs='+', help='list of classes')
    parser.add_argument('--TRAIN_TEST_SPLIT', type=float,
                        help='determines the proportion of data used for training vs testing')
    parser.add_argument('--CLASSIFICATION_LOSS')
    parser.add_argument('--REGRESSION_LOSS')
    parser.add_argument('--DATA_PATH')
    options = parser.parse_args()

    custom_dataset = data_generator.HandCommandsDataset(dataset_path=path)
    for i in range(len(custom_dataset)):
        image = visualize_single_instance(custom_dataset, i)

        plt.imshow(image.permute(1, 2, 0))
        plt.show()
