import os.path
import configargparse
import torch.utils.data
import torch
import data_generator
import paths


# partly inspired by: https://towardsdatascience.com/bounding-box-prediction-from-scratch-using-pytorch-a8525da51ddc

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
                 classes: list,
                 batch_size: int,
                 model: torch.nn.Module):
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
        self.classes = classes
        self.model = model

    def train_single_epoch(self, epoch_index):
        self.model.train()
        total = 0
        sum_loss = 0
        for img, label_tensor, label, instance_name in self.train_dataloader:
            batch = label.shape[0]
            img = img.cuda().float()
            label = label.cuda()
            label_tensor = label_tensor.cuda().float()
            out_class, out_pred = self.model(img)

            loss_class = torch.



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
                      classes=options.LABELS,
                      batch_size=5)
