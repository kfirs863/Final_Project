import torch
import pytorch_lightning as pl
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import models


class HandwritingClassifier(pl.LightningModule):
    def __init__(self, model, config, train_dataset, val_dataset):
        super(HandwritingClassifier, self).__init__()
        # self.model = models.efficientnet_b5(num_classes=config["model"]["num_classes"])
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Define loss function and metrics
        self.train_loss = CrossEntropyLoss()  # train loss function
        self.val_loss = CrossEntropyLoss()  # val loss function

        # Define metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.config["model"]["num_classes"],
                                                    compute_on_step=False)  # train accuracy metric
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.config["model"]["num_classes"],
                                                  compute_on_step=False)  # val accuracy metric

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["data"]["batch_size"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['data']['batch_size'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.train_loss(logit, y)
        acc = self.train_accuracy(logit, y)
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, logger=True)
        self.log("train_accuracy_step", acc, on_step=True, on_epoch=False, logger=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        acc = self.val_accuracy(logit, y)
        loss = self.val_loss(logit, y)
        self.log("val_loss_step", loss, on_step=True, on_epoch=True, logger=True)
        self.log("val_accuracy_step", acc, on_step=True, on_epoch=False, logger=True)
        return {"loss": loss, "acc": acc}

    def on_train_epoch_end(self):
        self.log("train_accuracy_epoch", self.train_accuracy.compute(), on_step=False, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        self.log("val_accuracy_epoch", self.val_accuracy.compute(), on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['module']['lr'])

        # Define the learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1, verbose=True),
            'interval': 'epoch',  # 'step' for step-wise learning rate decrease
            'frequency': 1,  # decrease lr every N epochs/steps
            'monitor': 'val_loss',  # metric to monitor for learning rate decrease
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
