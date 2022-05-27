import os.path
import configargparse
import matplotlib.pyplot as plt
import torch.utils.data
import torch
import code.datasets.data_generator as data_generator
import code.confs.paths as paths
import code.models.resnet34 as resnet34
from tqdm import tqdm
from code.utils.losses import losses
import code.utils.visualization as visualization

# partly inspired by: https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc


class Trainer(object):
    def __init__(self,
                 model,
                 optimizer,
                 batch_size,
                 classes,
                 train_dataset,
                 test_dataset,
                 classification_loss_fn,
                 regression_loss_fn):
        # TODO: IMPLEMENT CUSTOM LOSS FUNCTIONS IN TRAINING LOOP
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.classes = classes
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=self.batch_size)
        self.classification_loss_fn = classification_loss_fn
        self.regression_loss_fn = regression_loss_fn

    def train_single_epoch(self):
        self.model.train()
        total_instances_seen = 0
        sum_loss = 0
        for img, coord_tensor, label_tensor, instance_name in self.train_dataloader:
            # batch = label.shape[0]        # equivalent to self.batch_size
            img = img.cuda().float()
            label_tensor = label_tensor.cuda()
            coord_tensor = coord_tensor.cuda().float()
            # print(f"{img=}")
            # print(f"{label_tensor=}")
            print(f"{coord_tensor=}")
            print(f"{coord_tensor.shape=}")

            out_class, out_coord = self.model(img)

            out_coord = torch.sigmoid(out_coord)
            # print(f"{out_class=}")
            print(f"{out_coord=}")
            print(f"{out_coord.shape=}")

            out_bbox = visualization.center_tensor_to_bbox_tensor(out_coord)
            coord_bbox = visualization.center_tensor_to_bbox_tensor(coord_tensor)

            loss_class = self.classification_loss_fn(out_class, label_tensor)
            loss_coords = self.regression_loss_fn(out_bbox, coord_bbox)
            loss = loss_class + loss_coords
            # print(f"{loss_class=}")
            # print(f"{loss_coords=}")
            # print(f"{loss=}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_instances_seen += self.batch_size
            sum_loss += loss.item()
        train_loss = sum_loss/total_instances_seen
        return train_loss

    def val_metrics(self):
        self.model.eval()
        total_instances_seen = 0
        sum_loss = 0
        for img, coord_tensor, label_tensor, instance_name in self.test_dataloader:
            img = img.cuda().float()
            coord_tensor = coord_tensor.cuda()
            label_tensor = label_tensor.cuda().float()
            # print(f"{img=}")
            # print(f"{coord_tensor=}")
            # print(f"{label_tensor=}")

            out_class, out_coord = self.model(img)
            # print(f"{out_class=}")
            # print(f"{out_coord=}")

            loss_class = self.classification_loss_fn(out_class, label_tensor)
            loss_coords = self.regression_loss_fn(out_coord, coord_tensor)
            loss = loss_class + loss_coords
            # print(f"{loss_class=}")
            # print(f"{loss_coords=}")
            # print(f"{loss=}")
            sum_loss += loss.item()
            total_instances_seen += self.batch_size
        test_loss = sum_loss/total_instances_seen
        return test_loss

    def train_for(self, n_epochs):

        train_losses = []
        test_losses = []

        for i in tqdm(range(n_epochs)):
            train_loss = self.train_single_epoch()
            test_loss = self.val_metrics()

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        return train_losses, test_losses


if __name__ == '__main__':

    parser = configargparse.ArgumentParser(default_config_files=[os.path.join(paths.CONFS_PATH, "training.conf")])
    parser.add_argument('--TRAIN_TEST_SPLIT', type=float, help='determines the proportion of data used for training vs testing')
    parser.add_argument('--CLASSIFICATION_LOSS', type=str, help='determines the type of classification loss function used for training')
    parser.add_argument('--REGRESSION_LOSS', type=str, help='determines the type of regression loss function used for training')
    parser.add_argument('--DATA_PATH', type=str, help='the path where the dataset is stored')
    # TODO: add a model parse arg
    # TODO: add an optimizer parse arg

    options = parser.parse_args()
    print(f"{options=}")

    hand_commands_dataset = data_generator.HandCommandsDataset(dataset_path=os.path.abspath(options.DATA_PATH))
    train_split, test_split = data_generator.generate_dataset_splits(hand_commands_dataset, options.TRAIN_TEST_SPLIT)

    model = resnet34.BB_model(num_classes=len(hand_commands_dataset.label_list)).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.005)

    trainer = Trainer(train_dataset=train_split,
                      test_dataset=test_split,
                      classes=hand_commands_dataset.label_list,
                      batch_size=2,
                      model=model,
                      optimizer=optimizer,
                      classification_loss_fn=losses["classification"][options.CLASSIFICATION_LOSS],
                      regression_loss_fn=losses["regression"][options.REGRESSION_LOSS])

    train_L, test_L = trainer.train_for(20)

    fig, ax = plt.subplots()

    ax.plot(list(range(20)), train_L, label="TRAIN")
    ax.plot(list(range(20)), test_L, label="TEST")
    ax.legend()
    plt.show()

