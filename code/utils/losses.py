"""

This section describes and organises the potential loss functions we are considering for the task at hand.


For Label Classification:
Let N be the number of different labels, and B the Batch Size. Then 'input' and 'target' are defined by:

    - 'input': a [B, N] tensor, each row containing the probability of belonging to class 'n'.
        For example, with N = 3 and B = 2, 'input' could be:
                [[0.3, 0.4, 0.3],
                 [0.2, 0.7, 0.1]]

    - 'target': a [B, N] one hot encoded tensor, encoding the value of the label
        For example, with N = 3 and B = 2, 'target' could be:
                [[0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]]

The potential loss functions considered here are:
    - Cross Entropy: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
    - Binary Cross Entropy: https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html


For Regression / Localization of Sign:
Let B be the Batch Size. 'Input' and 'target' are defined by a [B, 4] tensor, rows are of the format (x_center, y_center, width, height)

The potential loss functions considered here are:
    - L1 loss on each component of the vectors: https://pytorch.org/docs/stable/generated/torch.nn.functional.l1_loss.html
    - GIoU loss on the bounding box corresponding to the vector: https://pytorch.org/vision/stable/generated/torchvision.ops.generalized_box_iou_loss.html

"""
import os.path
import torch
import torchvision
import code.utils.visualization as visualization
import code.datasets.data_generator as data_generator
import code.confs.paths as paths


def CE_loss(label_pred, label_truth):
    return torch.nn.functional.cross_entropy(label_pred, label_truth, reduction='mean')


def BCE_loss(label_pred, label_truth):
    return torch.nn.functional.binary_cross_entropy(label_pred, label_truth, reduction='mean')


def L1_loss(coord_pred, coord_truth):
    return torch.nn.functional.l1_loss(coord_pred, coord_truth, reduction='mean')


def GIoU_loss(coord_pred, coord_truth):
    truth_bbox = visualization.center_tensor_to_bbox_tensor(coord_truth)
    pred_bbox = visualization.center_tensor_to_bbox_tensor(coord_pred)
    return torchvision.ops.generalized_box_iou_loss(truth_bbox, pred_bbox, reduction='mean')


losses = {"Classification": {"CE": CE_loss,
                         "BCE": BCE_loss},
          "Regression": {"L1": L1_loss,
                         "GIoU": GIoU_loss}}


def evaluate_class_loss(loss_type, batch_size, n_classes, dataloader):
    classification_loss_fn = losses["Classification"][loss_type]

    print(f"SIMULATING OUTPUTS FOR CLASSIFICATION + LOSS: {loss_type}")
    for _, _, label_tensor, _ in dataloader:
        label_pred = torch.randn([batch_size, n_classes], requires_grad=True)
        label_pred = torch.softmax(label_pred, dim=1)

        classification_loss = classification_loss_fn(label_pred, label_tensor)
        print(f"loss = {classification_loss}")
        classification_loss.backward()

    label_tensor = torch.stack(batch_size * [torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)])
    perf_label_pred = torch.stack(batch_size * [torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)])
    perf_label_pred.requires_grad = True
    bad_label_pred = torch.stack(batch_size * [torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)])
    bad_label_pred.requires_grad = True

    print(f"For a ground truth tensor:\t\t\n{label_tensor}")
    print(f"Best possible prediction tensor:\t\n{perf_label_pred}")
    print(f"Worst possible prediction tensor:\t\n{bad_label_pred}")

    best_class_loss = classification_loss_fn(perf_label_pred, label_tensor)
    worst_class_loss = classification_loss_fn(bad_label_pred, label_tensor)
    print(f"Best possible loss score: {best_class_loss}")
    print(f"Worst possible loss score: {worst_class_loss}")
    best_class_loss.backward()
    worst_class_loss.backward()


def evaluate_regression_loss(loss_type, batch_size, dataloader):
    regression_loss_fn = losses["Regression"][loss_type]

    print(f"SIMULATING OUTPUTS FOR REGRESSION + LOSS: {loss_type}")
    for _, coord_tensor, _, _ in dataloader:
        coord_pred = visualization.random_bbox_tensor(B=batch_size, H=480, W=640)
        coord_pred = visualization.bbox_tensor_to_center_tensor(coord_pred)
        coord_pred.requires_grad = True

        regression_loss = regression_loss_fn(coord_pred, coord_tensor)
        print(f"loss = {regression_loss}")
        regression_loss.backward()

    image_pixel_resolution = (640, 480)
    box_width = 20
    box_height = 30
    coord_tensor = torch.stack(batch_size * [torch.tensor([box_width / 2,
                                                  box_height / 2,
                                                  box_width,
                                                  box_height], dtype=torch.float32)])
    perf_coord_pred = torch.stack(batch_size * [torch.tensor([box_width / 2,
                                                     box_height / 2,
                                                     box_width,
                                                     box_height], dtype=torch.float32)])
    perf_coord_pred.requires_grad = True
    bad_coord_pred = torch.stack(batch_size * [torch.tensor([image_pixel_resolution[0] - box_width / 2,
                                                    image_pixel_resolution[1] - box_height / 2,
                                                    box_width,
                                                    box_height], dtype=torch.float32)])
    bad_coord_pred.requires_grad = True

    print(f"For a ground truth tensor:\t\t\n{coord_tensor}")
    print(f"Best possible prediction tensor:\t\n{perf_coord_pred}")
    print(f"Worst possible prediction tensor:\t\n{bad_coord_pred}")

    best_coord_loss = regression_loss_fn(perf_coord_pred, coord_tensor)
    worst_coord_loss = regression_loss_fn(bad_coord_pred, coord_tensor)
    print(f"\nBest possible loss score: {best_coord_loss}")
    print(f"Worst possible loss score: {worst_coord_loss}")
    best_coord_loss.backward()
    worst_coord_loss.backward()


if __name__ == '__main__':
    """
    An example of loss comparisons between random inputs and outputs,
    with shape and format that will be used during regular training.
    """
    B = 1       # batch size

    instances_to_check = [0, 1, 2, 3, 4, 5, 6, 7]
    path = os.path.join(paths.DATA_PATH, "annotated")
    dataset = data_generator.HandCommandsDataset(path)
    subset = torch.utils.data.Subset(dataset, instances_to_check)
    dataloader = torch.utils.data.DataLoader(subset, batch_size=B, shuffle=True)

    class_loss_type = "CE"
    evaluate_class_loss(loss_type=class_loss_type, batch_size=B, n_classes=len(dataset.label_list), dataloader=dataloader)
    print()

    class_loss_type = "BCE"
    evaluate_class_loss(loss_type=class_loss_type, batch_size=B, n_classes=len(dataset.label_list), dataloader=dataloader)
    print()

    regr_loss_type = "L1"
    evaluate_regression_loss(loss_type=regr_loss_type, batch_size=B, dataloader=dataloader)
    print()

    regr_loss_type = "GIoU"
    evaluate_regression_loss(loss_type=regr_loss_type, batch_size=B, dataloader=dataloader)
    print()
