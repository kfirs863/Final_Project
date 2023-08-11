import logging

import comet_ml
import pytorch_lightning as pl
import torch
import torchvision
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor

from src.models import CNNModel, ResNet101

from pytorch_lightning.loggers import CometLogger
from src.lightning_module import HandwritingClassifier
from torchvision.datasets import ImageFolder
from torchvision import transforms

from src.utils.custom_transformer import SplitAndStackImageToSquare

# Create a dictionary mapping model names to classes
model_dict = {
    'CNNModel': CNNModel,
    'ResNet101': ResNet101,
    'EfficientNetB4': torchvision.models.efficientnet_b4,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(config,experiment=None):

    # get std and mean from config
    std = config["data"]["std"]
    mean = config["data"]["mean"]

    IMAGE_SIZE = config["data"]["image_size"]
    # Init our data pipeline, model and lightning module
    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        SplitAndStackImageToSquare(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomInvert(p=0.25),
        transforms.ToTensor(),  # Convert image to PyTorch Tensor
        # transforms.Normalize(mean,std),
    ])

    # If parameters provided, update the config with optimizer parameters
    if experiment:
        config["model"]["learning_rate"] = experiment.get_parameter("learning_rate")
        config["model"]["batch_size"] = experiment.get_parameter("batch_size")

    # datasets
    train_dataset = ImageFolder(config["data"]["train_data_path"], transform=transform)
    val_dataset = ImageFolder(config["data"]["val_data_path"], transform=transform)
    test_dataset = ImageFolder(config["data"]["test_data_path"], transform=transform)

    # Retrieve model class from dictionary
    ModelClass = model_dict[config["model"]["model_name"]]

    # Initialize model
    model = ModelClass(num_classes=config["model"]["num_classes"])


    lightning_module = HandwritingClassifier(model, config, train_dataset, val_dataset,test_dataset)

    comet_logger = CometLogger(
        api_key="jsPqM9osr1ZfIKWiEeiAlitCa",
        workspace="final-project",
        project_name="hand-writing-classification",
        distributed_mode='client',
    )

    # log config dict to comet as assets
    comet_logger.experiment.log_asset_data(config)

    # log model class source code to comet as assets
    comet_logger.experiment.log_asset("src/models/cnn_model.py")

    # Log total number of model parameters
    comet_logger.experiment.log_metric("num_model_parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Define a LearningRateMonitor callback
    lr_monitor = LearningRateMonitor()

    # Init trainer
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        logger=comet_logger,
        callbacks=[lr_monitor],
        profiler="Simple",
        accelerator='ddp',
        precision=config["trainer"]["precision"],
    )

    # Fit model
    trainer.fit(lightning_module)
    comet_logger.experiment.end()

if __name__ == "__main__":
    import os
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize optimizer using the configuration
    # opt = Optimizer("optimizer.yaml", project_name="optimizer-search-01")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,5,6,7"
    main(config)
