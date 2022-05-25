"""

This section describes and organises the potential loss functions we are considering for the task at hand.


For Label Classification:
Let N be the number of different labels, and B the Batch Size. then 'input' and 'target' are defined by:

    - 'input': a [B, N] tensor, each row containing the probability of belonging to class 'n'.
        For example, with N = 3 and B = 2, 'input' could be:
                [[0.3, 0.4, 0.3],
                 [0.2, 0.7, 0.1]]

    - 'target': a [B, N] one hot encoded tensor, encoding the value of the label
        For example, with N = 3 and B = 2, 'target' could be:
                [[0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]

The potential loss functions considered here are:
    - Cross Entropy: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
    - Binary Cross Entropy: https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html#torch.nn.functional.binary_cross_entropy


For Regression / Localization of Sign:


"""
import os.path
import torch
import torchvision
import code.utils.visualization as visualization
import code.datasets.data_generator as data_generator
import code.confs.paths as paths


def GIoU_loss(coord_pred, coord_truth):
    truth_bbox = visualization.center_tensor_to_bbox_tensor(coord_truth)
    pred_bbox = visualization.center_tensor_to_bbox_tensor(coord_pred)
    return torchvision.ops.generalized_box_iou_loss(truth_bbox, pred_bbox, reduction='mean')


losses = {"Classification": {"CE": torch.nn.functional.cross_entropy,
                         "BCE": torch.nn.functional.binary_cross_entropy},
          "Regression": {"L1": torch.nn.functional.l1_loss,
                         "GIoU": GIoU_loss}}


if __name__ == '__main__':
    """
    An example of loss comparisons between random inputs and outputs,
    with shape and format that will be used during regular training.
    """
    B = 5       # Batch Size
    classification_loss_fn = losses["Classification"]["CE"]
    regression_loss_fn = losses["Regression"]["GIoU"]
    instances_to_check = [0, 1, 2, 3, 4, 5, 6, 7]

    path = os.path.join(paths.DATA_PATH, "annotated")
    dataset = data_generator.HandCommandsDataset(path)
    subset = torch.utils.data.Subset(dataset, instances_to_check)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)


    print("SIMULATING OUTPUTS FOR CLASSIFICATION + LOSS\n")
    for img, coord_tensor, label_tensor, _ in dataloader:

        label_pred = torch.randn([B, len(dataset.label_list)], requires_grad=True)
        label_pred = torch.softmax(label_pred, dim=1)

        classification_loss = classification_loss_fn(label_pred, label_tensor)
        print(f"loss = {classification_loss}")
        classification_loss.backward()

    print("\n\nSIMULATING OUTPUTS FOR REGRESSION + LOSS\n\n")
    for img, coord_tensor, label_tensor, _ in dataloader:

        coord_pred = visualization.random_bbox_tensor(B=B, H=480, W=640)
        coord_pred = visualization.bbox_tensor_to_center_tensor(coord_pred)
        coord_pred.requires_grad = True

        regression_loss = regression_loss_fn(coord_pred, coord_tensor)
        print(f"loss = {regression_loss}")
        regression_loss.backward()
