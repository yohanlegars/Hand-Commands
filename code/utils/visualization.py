import torch
import torchvision
import data_generator
import paths
import os
import matplotlib.pyplot as plt


def label_tensor_to_bbox_tensor(label_tensor):
    """
    :param label_tensor: tensor([x, y, width, height])
    :return: tensor([x_min, y_min, x_max, y_max])
    """
    x_min = int(label_tensor[0] - 0.5 * label_tensor[2])
    x_max = int(label_tensor[0] + 0.5 * label_tensor[2])
    y_min = int(label_tensor[1] - 0.5 * label_tensor[3])
    y_max = int(label_tensor[1] + 0.5 * label_tensor[3])
    return torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.int)


def visualize_single_instance(idx):
    """
    :param idx: index of the instance
    :return: None
    """
    path = os.path.join(paths.DATA_PATH, "annotated")
    dataset = data_generator.HandCommandsDataset(path)
    image, labeltensor, label, _ = dataset.__getitem__(idx)
    bbox = label_tensor_to_bbox_tensor(labeltensor)
    label = [label]
    bbox = torch.reshape(bbox, (1, -1))
    visual = torchvision.utils.draw_bounding_boxes(image=image, boxes=bbox, labels=label)
    plt.imshow(visual.permute(1, 2, 0))
    plt.show()


if __name__ == '__main__':
    visualize_single_instance(5)
