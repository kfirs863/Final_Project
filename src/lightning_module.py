import torch
import pytorch_lightning as pl
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader


class HandwritingClassifier(pl.LightningModule):
    def __init__(self, model, config, train_dataset, val_dataset,test_dataset=None):
        super(HandwritingClassifier, self).__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Define loss function and metrics
        self.train_loss = CrossEntropyLoss()  # train loss function
        self.val_loss = CrossEntropyLoss()  # val loss function
        self.test_loss = CrossEntropyLoss()  # test loss function

        # Define metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.config["model"]["num_classes"])  # train accuracy metric
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.config["model"]["num_classes"])  # val accuracy metric
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.config["model"]["num_classes"])  # test accuracy metric

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config["data"]["batch_size"], shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config['data']['batch_size'], shuffle=False, num_workers=8)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.train_loss(logit, y)
        self.train_accuracy(logit, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.val_loss(logit, y)
        self.val_accuracy(logit, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_accuracy", self.train_accuracy.compute(), on_step=False, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        self.log("val_accuracy", self.val_accuracy.compute(), on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.test_loss(logit, y)
        test_accuracy = self.test_accuracy(logit, y)  # compute accuracy for this batch
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("test_accuracy", test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss, "test_accuracy": test_accuracy}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_accuracy"] for x in outputs]).mean()
        self.log("avg_test_loss", avg_loss, prog_bar=True, logger=True)
        self.log("test_accuracy", self.test_accuracy.compute(), on_step=False, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['module']['learning_rate'])

        # Define the learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=6,
                                                                    verbose=True,threshold=1e-3,min_lr=1e-5),
            'monitor': 'train_accuracy',  # metric to monitor for learning rate decrease
        }

        return [optimizer]#, [scheduler]
