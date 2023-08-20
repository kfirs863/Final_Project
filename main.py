import os
import time
import comet_ml
from pathlib import Path
from collections import Counter

import pytorch_lightning as pl
import torchvision
import yaml
import plotly.express as px
import pandas as pd

from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.lightning_module import HandwritingClassifier
from src.models import CNNModel, ResNet101
from src.utils.custom_transformer import SplitAndStackImageToSquare

# Create a dictionary mapping model names to classes
model_dict = {
    'CNNModel': CNNModel,
    'ResNet101': ResNet101,
    'EfficientNetB4': torchvision.models.efficientnet_b4,
}


def main(config):

    # Set image size
    IMAGE_SIZE = config["data"]["image_size"]

    # Common preprocessing transformations
    valid_and_test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        SplitAndStackImageToSquare(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # Training transformations with data augmentation
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        SplitAndStackImageToSquare(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandAugment(),
        transforms.ToTensor(),
    ])

    # datasets
    train_dataset = ImageFolder(config["data"]["train_data_path"], transform=train_transforms)
    val_dataset = ImageFolder(config["data"]["val_data_path"], transform=valid_and_test_transforms)
    test_dataset = ImageFolder(config["data"]["test_data_path"], transform=valid_and_test_transforms)

    # Retrieve model class from dictionary
    ModelClass = model_dict[config["model"]["model_name"]]

    # Initialize model
    model = ModelClass(num_classes=config["model"]["num_classes"],image_size=IMAGE_SIZE)

    # Initialize lightning module
    lightning_module = HandwritingClassifier(model, config, train_dataset, val_dataset, test_dataset)

    # Define a CometLogger callback
    comet_logger = CometLogger(
        api_key="jsPqM9osr1ZfIKWiEeiAlitCa",
        workspace="final-project",
        project_name="hand-writing-classification",
        save_dir='/homes/kfirs/PycharmProjects/FinalProject/comet_logs/'
        # experiment_name="Testing_dataset_203_classes_last_layer_removed",
    )

    # Define a LearningRateMonitor callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # define checkpoint every time 'val_accuarcy' has improved
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_accuracy',
        filename='best-{epoch:02d}-{val_accuracy:.2f}',
        save_top_k=1,
        mode='max',
    )

    # Init trainer
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        logger=comet_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        profiler="Simple",
        log_every_n_steps=config["trainer"]["log_every_n_steps"],
        precision=config["trainer"]["precision"],
    )


    # log_class_distribution for each dataset
    list(map(lambda dataset: log_class_distribution(dataset, comet_logger), [train_dataset, val_dataset, test_dataset]))

    # log config dict to comet as assets
    comet_logger.experiment.log_asset_data(config)

    # sleep for 1 second to allow comet to sync
    time.sleep(1)

    comet_logger.log_hyperparams(config)

    # sleep for 1 second to allow comet to sync
    time.sleep(1)

    # log model class source code to comet as assets
    comet_logger.experiment.log_code("src/models/cnn_model.py")

    # log lightning module source code to comet as assets
    comet_logger.experiment.log_code('./src/lightning_module.py')

    # Log total number of model parameters
    comet_logger.experiment.log_metric("num_model_parameters",
                                       sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Fit model
    trainer.fit(lightning_module)

    # Log best model to Comet ML
    for checkpoint_path in checkpoint_callback.best_k_models.keys():
        comet_logger.experiment.log_asset(checkpoint_path)

    # Evaluate model
    trainer.test(lightning_module)

    # End Comet ML experiment
    comet_logger.experiment.end()


def log_class_distribution(dataset: ImageFolder, comet_logger):
    """
    Log class distribution for a PyTorch ImageFolder dataset using Plotly.
    :param dataset:  dataset to calculate class distribution for
    :param comet_logger: comet logger
    :return: None
    """

    # Get the class labels from the dataset
    class_counts = Counter(dataset.targets)

    # Convert class indices to class labels
    class_labels_counts = {dataset.classes[idx]: count for idx, count in class_counts.items()}

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'Class': [str(c) for c in class_labels_counts.keys()],
        'Count': list(class_labels_counts.values())
    })

    # Sort the DataFrame by class labels as integers
    df['Class'] = df['Class'].astype(int)  # Convert to integers
    df = df.sort_values('Class')

    # Create a bar chart using Plotly
    fig = px.bar(df, x='Class', y='Count', title=f"{Path(dataset.root).name.capitalize()} Class Distribution")

    # Update x-axis to show labels more frequently
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=10)

    # Log the HTML file as an asset to Comet ML
    comet_logger.experiment.log_html(fig.to_html())


if __name__ == "__main__":

    # Set COMET_GIT_DIRECTORY environment variable
    os.environ["COMET_GIT_DIRECTORY"] = str('/homes/kfirs/PycharmProjects/FinalProject')

    # Load config file
    with open("develop_config.yaml", "r") as f:
        config = yaml.safe_load(f)


    main(config)
