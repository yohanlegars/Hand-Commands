import torch
import torchvision
import code.datasets.data_generator as data_generator
import code.confs.paths as paths
import os
import matplotlib.pyplot as plt
import configargparse


def coord_tensor_to_bbox_tensor(label_tensor):
    """
    :param label_tensor: tensor([x, y, width, height])
    :return: tensor([x_min, y_min, x_max, y_max])
    """
    x_min = int(label_tensor[0] - 0.5 * label_tensor[2])
    x_max = int(label_tensor[0] + 0.5 * label_tensor[2])
    y_min = int(label_tensor[1] - 0.5 * label_tensor[3])
    y_max = int(label_tensor[1] + 0.5 * label_tensor[3])
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float32)


def label_tensor_to_string(label_tensor: torch.Tensor, label_list):
    label_tensor = label_tensor.clone().detach()
    label_tensor = label_tensor.int()
    label_tensor = label_tensor.argmax(-1)
    return label_tensor


def visualize_single_instance(dataset, idx):
    """
    :param idx: index of the instance
    :return: None
    """
    #TODO: find a nice way to display labels
    image, coord_tensor, label_tensor, _ = dataset.__getitem__(idx)
    bbox = coord_tensor_to_bbox_tensor(coord_tensor)
    label_tensor = label_tensor_to_string(label_tensor, dataset.label_list)
    bbox = torch.reshape(bbox, (1, -1))

    visual = torchvision.utils.draw_bounding_boxes(image=image, boxes=bbox, labels=[""])
    plt.imshow(visual.permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    path = os.path.join(paths.DATA_PATH, "annotated")

    parser = configargparse.ArgumentParser(default_config_files=[os.path.join(paths.CONFS_PATH, "training.conf")])
    parser.add_argument('--LABELS', type=str, nargs='+', help='list of classes')
    parser.add_argument('--TRAIN_TEST_SPLIT', type=float,
                        help='determines the proportion of data used for training vs testing')
    options = parser.parse_args()

    custom_dataset = data_generator.HandCommandsDataset(dataset_path=path)
    for i in range(len(custom_dataset)):
        visualize_single_instance(custom_dataset, i)
