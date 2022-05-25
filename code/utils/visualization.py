import torch
import torchvision
import code.datasets.data_generator as data_generator
import code.confs.paths as paths
import os
import matplotlib.pyplot as plt
import configargparse


def coord_tensor_to_bbox_tensor(coord_tensor):
    """
    This function can be used to convert a coordinate tensor, into its corresponding bounding box tensor

    :param coord_tensor: tensor([x, y, width, height])
    :return: tensor([x_min, y_min, x_max, y_max])
    """
    x_min = int(coord_tensor[0] - 0.5 * coord_tensor[2])
    x_max = int(coord_tensor[0] + 0.5 * coord_tensor[2])
    y_min = int(coord_tensor[1] - 0.5 * coord_tensor[3])
    y_max = int(coord_tensor[1] + 0.5 * coord_tensor[3])
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)


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
    bbox = coord_tensor_to_bbox_tensor(coord_tensor)
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
    options = parser.parse_args()

    custom_dataset = data_generator.HandCommandsDataset(dataset_path=path)
    for i in range(len(custom_dataset)):
        image = visualize_single_instance(custom_dataset, i)

        plt.imshow(image.permute(1, 2, 0))
        plt.show()
