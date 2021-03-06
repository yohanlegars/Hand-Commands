import json
import os.path
import uuid

import configargparse
import matplotlib.pyplot as plt
import torch.utils.data
import torch
import code.datasets.data_generator as data_generator
import code.confs.paths as paths
from tqdm import tqdm
from code.utils.losses import losses
from code.utils.models_catalog import models
from datetime import datetime

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
                 regression_loss_fn,
                 save_path):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.classes = classes
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            drop_last=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                           batch_size=self.batch_size,
                                                           drop_last=True)
        self.classification_loss_fn = classification_loss_fn
        self.regression_loss_fn = regression_loss_fn
        self.save_path = save_path

    def train_single_epoch(self):
        self.model.train()
        total_instances_seen = 0
        sum_loss = 0
        for input_img, coord_truth, label_truth, instance_name in self.train_dataloader:
            input_img = input_img.cuda().float()
            label_truth = label_truth.cuda()
            coord_truth = coord_truth.cuda().float()

            label_pred, coord_pred = self.model(input_img)
            label_pred = torch.nn.functional.softmax(label_pred, dim=-1)

            loss_class = self.classification_loss_fn(label_pred, label_truth)
            loss_coords = self.regression_loss_fn(coord_pred, coord_truth)
            loss = loss_class + loss_coords

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
        for input_img, coord_truth, label_truth, instance_name in self.test_dataloader:
            input_img = input_img.cuda().float()
            coord_truth = coord_truth.cuda()
            label_truth = label_truth.cuda().float()

            label_pred, coord_pred = self.model(input_img)
            label_pred = torch.nn.functional.softmax(label_pred, dim=-1)

            loss_class = self.classification_loss_fn(label_pred, label_truth)
            loss_coords = self.regression_loss_fn(coord_pred, coord_truth)
            loss = loss_class + loss_coords

            total_instances_seen += self.batch_size
            sum_loss += loss.item()
        test_loss = sum_loss/total_instances_seen
        return test_loss

    def train_for(self, n_epochs, save=True):

        train_losses = []
        test_losses = []

        for epoch_idx in tqdm(range(n_epochs)):
            train_loss = self.train_single_epoch()
            test_loss = self.val_metrics()

            train_losses.append(train_loss)
            test_losses.append(test_loss)

        if save:
            self.save_model()

        return train_losses, test_losses

    def save_model(self):

        model_name = f"{type(self.model).__name__}_{datetime.now().strftime('%d-%m-%Y_%H-%M')}_{uuid.uuid1()}"

        torch.save(self.model.state_dict(), os.path.join(self.save_path, model_name))


if __name__ == '__main__':

    parser = configargparse.ArgumentParser(default_config_files=[os.path.join(paths.CONFS_PATH, "config_templates/training.conf")])
    parser.add_argument('--TRAIN_TEST_SPLIT', type=float, help='determines the proportion of data used for training vs testing')
    parser.add_argument('--CLASSIFICATION_LOSS', type=str, help='determines the type of classification loss function used for training')
    parser.add_argument('--REGRESSION_LOSS', type=str, help='determines the type of regression loss function used for training')
    parser.add_argument('--DATA_PATH', type=str, help='the path where the dataset is stored')
    parser.add_argument('--MODEL', type=str, help='the model chosen for training')
    parser.add_argument('--OPTIMIZER', type=str, help='the optimizer chosen for surfing on the loss function')
    parser.add_argument('--N_EPOCHS', type=int, help='number of epochs to train for')
    parser.add_argument('--MODELS_PATH', type=str, help='the path where the models should be saved')
    parser.add_argument('--N_AUGMENTS', type=int, help='the number of times to perform random augmentations on instances. artificially increases the size of the dataset.')
    parser.add_argument('--BATCH_SIZE', type=int, help='the batch size')
    # parser.add_argument('--CONF',type=str, default='./confs/training.conf')

    options, _ = parser.parse_known_args()

    print(f"{options=}")

    print("Beginning Training Process with these configurations:")
    print(json.dumps(vars(options), indent=4))

    hand_commands_dataset = data_generator.HandCommandsDataset(dataset_path=os.path.abspath(options.DATA_PATH))
    train_split, test_split = data_generator.generate_dataset_splits(hand_commands_dataset, options.TRAIN_TEST_SPLIT)

    model = models[options.MODEL]
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = eval("torch.optim.{}(parameters)".format(options.OPTIMIZER))

    trainer = Trainer(train_dataset=train_split,
                      test_dataset=test_split,
                      classes=hand_commands_dataset.label_list,
                      batch_size=options.BATCH_SIZE,
                      model=model,
                      optimizer=optimizer,
                      classification_loss_fn=losses["classification"][options.CLASSIFICATION_LOSS],
                      regression_loss_fn=losses["regression"][options.REGRESSION_LOSS],
                      save_path=os.path.abspath(options.MODELS_PATH))

    train_L, test_L = trainer.train_for(options.N_EPOCHS)

    fig, ax = plt.subplots()

    ax.plot(list(range(options.N_EPOCHS)), train_L, label="TRAIN")
    ax.plot(list(range(options.N_EPOCHS)), test_L, label="TEST")
    ax.legend()
    plt.show()

