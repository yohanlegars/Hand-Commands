import os.path
import configargparse
import matplotlib.pyplot as plt
import torch.utils.data
import torch
import code.datasets.data_generator as data_generator
import code.confs.paths as paths
import code.models.resnet34 as resnet34
from tqdm import tqdm


# partly inspired by: https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc

def generate_dataset_splits(dataset, splitratio):
    split_one_amount = int(splitratio * len(dataset))
    split_two_amount = len(dataset) - split_one_amount
    assert len(dataset) == split_one_amount + split_two_amount

    split_one, split_two = torch.utils.data.random_split(dataset, [split_one_amount, split_two_amount])
    return split_one, split_two


class Trainer(object):
    def __init__(self,
                 train_dataset: data_generator.HandCommandsDataset,
                 test_dataset: data_generator.HandCommandsDataset,
                 classes: list,
                 batch_size: int,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 C: int=1000):
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=self.batch_size)
        self.classes = classes
        self.model = model
        self.C = C
        self.optimizer = optimizer

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
            # print(f"{coord_tensor=}")

            out_class, out_coord = self.model(img)
            # print(f"{out_class=}")
            # print(f"{out_coord=}")

            loss_class = torch.nn.functional.cross_entropy(out_class, label_tensor)
            loss_coords = torch.nn.functional.l1_loss(out_coord, coord_tensor)
            loss = loss_class + loss_coords / self.C
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

            loss_class = torch.nn.functional.cross_entropy(out_class, label_tensor)
            loss_coords = torch.nn.functional.l1_loss(out_coord, coord_tensor)
            loss = loss_class + loss_coords / self.C
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

    path = os.path.join(paths.DATA_PATH, "annotated")

    parser = configargparse.ArgumentParser(default_config_files=[os.path.join(paths.CONFS_PATH, "training.conf")])
    parser.add_argument('--TRAIN_TEST_SPLIT', type=float, help='determines the proportion of data used for training vs testing')

    options = parser.parse_args()

    hand_commands_dataset = data_generator.HandCommandsDataset(dataset_path=path)

    train_split, test_split = generate_dataset_splits(hand_commands_dataset, options.TRAIN_TEST_SPLIT)

    model = resnet34.BB_model(num_classes=len(hand_commands_dataset.label_list)).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.005)

    trainer = Trainer(train_dataset=train_split,
                      test_dataset=test_split,
                      classes=hand_commands_dataset.label_list,
                      batch_size=2,
                      model=model,
                      optimizer=optimizer)

    train_L, test_L = trainer.train_for(20)

    fig, ax = plt.subplots()

    ax.plot(list(range(20)), train_L, label="TRAIN")
    ax.plot(list(range(20)), test_L, label="TEST")
    ax.legend()
    plt.show()

