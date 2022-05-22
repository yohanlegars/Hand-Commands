import torch.nn
import torch.utils.data

class Trainer():
    def __init__(self,
                 model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 loss_fn: torch.nn.Module,
                 optimizer: torch.optim.Optimizer):
        self.model = model
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

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



