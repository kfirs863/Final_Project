import torch
import pytorch_lightning as pl
import torchmetrics
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import seaborn as sns


class HandwritingClassifier(pl.LightningModule):
    def __init__(self, model, config, train_dataset=None, val_dataset=None, test_dataset=None):
        super(HandwritingClassifier, self).__init__()
        self.model = model
        self.config = config
        self.batch_size = config["data"]["batch_size"]
        self.num_classes = self.config["model"]["num_classes"]
        self.save_hyperparameters(ignore=['model'])

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Define loss function and metrics
        self.train_loss = CrossEntropyLoss()  # train loss function
        self.val_loss = CrossEntropyLoss()  # val loss function
        self.test_loss = CrossEntropyLoss()  # test loss function

        # Define metrics
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                   num_classes=self.num_classes)  # train accuracy metric
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                 num_classes=self.num_classes)  # val accuracy metric
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass",
                                                                  num_classes=self.num_classes)  # test accuracy metric

        # Define confusion matrix metric for test
        self.confusion_matrix_metric = ConfusionMatrix(num_classes=self.num_classes, task="multiclass")

        # Extract class names from ImageFolder
        if isinstance(test_dataset, ImageFolder):
            self.class_names = test_dataset.classes
        else:
            self.class_names = None  # Handle case where train_dataset is not an ImageFolder

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.train_loss(logit, y)
        self.train_accuracy(logit, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.val_loss(logit, y)
        self.val_accuracy(logit, y)
        self.log("val_loss", loss, on_step=True, on_epoch=False, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, logger=True)

    def on_validation_epoch_end(self):
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logit = self(x)
        loss = self.test_loss(logit, y)
        self.test_accuracy(logit, y)
        self.confusion_matrix_metric.update(logit, y)
        self.log("test_loss", loss, on_step=True, on_epoch=False, logger=True)
        return loss

    def on_test_epoch_end(self):
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True, logger=True)

        confusion_matrix = self.confusion_matrix_metric.compute().cpu().numpy()
        # if self.class_names:
        #     self.logger.experiment.log_confusion_matrix(labels=self.class_names, matrix=confusion_matrix,
        #                                                 max_categories=len(self.class_names))
        # else:
        #     self.logger.experiment.log_confusion_matrix(matrix=confusion_matrix)  # Without class names

        # Slice the confusion matrix and class names to include only the first 30 classes
        confusion_matrix = confusion_matrix[:30, :30]
        class_names_subset = self.class_names[:30]

        # Show confusion matrix using matplotlib
        fig, ax = plt.subplots(figsize=(15, 15))  # Adjust the figure size
        sns.heatmap(confusion_matrix, annot=True, ax=ax, cmap="Blues", fmt="d",
                    annot_kws={"size": 12})  # Set annotation size

        ax.set_xlabel("Predicted Labels", fontsize=14)  # Set x-axis label size
        ax.set_ylabel("True Labels", fontsize=14)  # Set y-axis label size
        ax.set_title("Confusion Matrix", fontsize=16)  # Set title size

        if class_names_subset:
            ax.xaxis.set_ticklabels(class_names_subset, fontsize=12)  # Set x-axis tick label size
            ax.yaxis.set_ticklabels(class_names_subset, fontsize=12)  # Set y-axis tick label size

        plt.show()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['module']['learning_rate'])

        # Define the learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.config['scheduler']['mode'],
                                                                    factor=self.config['scheduler']['factor'],
                                                                    patience=self.config['scheduler']['patience'],
                                                                    verbose=True,
                                                                    threshold=float(
                                                                        self.config['scheduler']['threshold']),
                                                                    min_lr=float(self.config['scheduler']['min_lr'])),
            'monitor': self.config['scheduler']['monitor'],  # metric to monitor for learning rate decrease
        }

        return [optimizer], [scheduler]
