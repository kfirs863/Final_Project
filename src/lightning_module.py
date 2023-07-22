import torch
import pytorch_lightning as pl
import torchmetrics
from torch.nn import CrossEntropyLoss
from torchvision import models

class HandwritingClassifier(pl.LightningModule):
    def __init__(self, model, config):
        super(HandwritingClassifier, self).__init__()
        self.model = models.efficientnet_b5(num_classes=config["model"]["num_classes"])
        self.config = config
        self.save_hyperparameters(ignore=['model'])

        # Define loss function and metrics
        self.train_loss = CrossEntropyLoss()  # train loss function
        self.val_loss = CrossEntropyLoss()  # val loss function

        # Define metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=self.config["model"]["num_classes"],compute_on_step=False)  # train accuracy metric
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass",num_classes=self.config["model"]["num_classes"],compute_on_step=False)  # val accuracy metric

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.train_loss(logit, y)
        self.train_accuracy(logit, y)
        self.log("train_loss_step", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        self.val_accuracy(logit, y)
        loss = self.val_loss(logit, y)
        self.log("val_loss_step", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_accuracy_epoch",self.train_accuracy.compute(),on_step=False, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        self.log("val_accuracy_epoch", self.val_accuracy.compute(),on_step=False, on_epoch=True, logger=True)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['module']['lr'])
        return optimizer
