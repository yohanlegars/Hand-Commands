import os.path
import configargparse
import torch.utils.data
import torch
import data_generator
import paths


def generate_dataset_splits(dataset, splitratio):
    split_one_amount = int(splitratio * len(dataset))
    split_two_amount = len(dataset) - split_one_amount
    assert len(dataset) == split_one_amount + split_two_amount

    split_one, split_two = torch.utils.data.random_split(dataset, [split_one_amount, split_two_amount])
    return split_one, split_two


class Trainer(object):
    def __init__(self,
                 train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset,
                 classes: list):

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.classes = classes

    def train_one_epoch(self, epoch_index):
        running_loss = 0.
        last_loss = 0.

        for idx, data in enumerate(self.data_loader):

            # retrieve the image and annotation dictionary from the training instance
            img, annot_dict = data

            # zero the gradients for the batch
            self.optimizer.zero_grad()

            # prediction / forward pass     # TODO: see how it works with formatting of output
            outputs = self.model(img)

            # compute loss and gradients
            loss = self.loss_fn(outputs, annot_dict)

            # parameter update
            self.optimizer.step()


if __name__ == '__main__':

    path = os.path.join(paths.DATA_PATH, "annotated")
    hand_commands_dataset = data_generator.HandCommandsDataset(path)

    parser = configargparse.ArgumentParser(default_config_files=[os.path.join(paths.CONFS_PATH, "training.conf")])
    parser.add_argument('--LABELS', type=str, nargs='+', help='list of classes')
    parser.add_argument('--TRAIN_TEST_SPLIT', type=float, help='determines the proportion of data used for training vs testing')

    options = parser.parse_args()

    train_split, test_split = generate_dataset_splits(hand_commands_dataset, options.TRAIN_TEST_SPLIT)

    trainer = Trainer(train_dataset=train_split,
                      test_dataset=test_split,
                      classes=options.LABELS)
