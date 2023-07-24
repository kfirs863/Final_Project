import comet_ml
import pytorch_lightning as pl
import torch
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
    'ResNet101': ResNet101
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(config):
    comet_logger = CometLogger(
        api_key="jsPqM9osr1ZfIKWiEeiAlitCa",
        workspace="final-project",
        project_name="Hand_Writing_Classification",
        experiment_name="CNN Experiment StepLR 5"
    )

    # Init our data pipeline, model and lightning module
    # Define transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        SplitAndStackImageToSquare(),
        transforms.Resize((890, 890)),
        transforms.ToTensor(),  # Convert image to PyTorch Tensor
    ])

    # datasets
    train_dataset = ImageFolder(config["data"]["train_data_path"], transform=transform)
    val_dataset = ImageFolder(config["data"]["val_data_path"], transform=transform)

    # Retrieve model class from dictionary
    ModelClass = model_dict[config["model"]["model_name"]]

    # Initialize model
    model = ModelClass(config["model"]["num_classes"])

    lightning_module = HandwritingClassifier(model, config, train_dataset, val_dataset)

    # Define a LearningRateMonitor callback
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Init trainer
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        logger=comet_logger,
        log_every_n_steps=config["trainer"]["log_every_n_steps"],
        callbacks=[lr_monitor],
        precision=config["trainer"]["precision"],
    )

    # Fit model
    trainer.fit(lightning_module)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    main(config)
