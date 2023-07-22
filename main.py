import comet_ml
import pytorch_lightning as pl
import torch
import yaml
from src.models import CNNModel

from pytorch_lightning.loggers import CometLogger
from src.lightning_module import HandwritingClassifier
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# Create a dictionary mapping model names to classes
model_dict = {
    'CNNModel': CNNModel,
    # add more models as you define them
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(config):
    # Comet.ml logger
    comet_logger = CometLogger(
        api_key="jsPqM9osr1ZfIKWiEeiAlitCa",
        workspace="final-project",
        project_name="Hand_Writing_Classification",
        experiment_name="efficientnet_b5 Experiment 1"
    )

    # Init our data pipeline, model and lightning module
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((160,4964)),
        # transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert image to PyTorch Tensor
        transforms.Lambda(lambda x: torch.rot90(x, 1, [1,2])),
    ])

    # datasets
    train_dataset = ImageFolder(config["data"]["train_data_path"], transform=transform)
    val_dataset = ImageFolder(config["data"]["val_data_path"], transform=transform)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False)

    print("Train dataset shape: ", train_dataset[0][0].shape)
    print("Val dataset shape: ", val_dataset[0][0].shape)

    # Retrieve model class from dictionary
    ModelClass = model_dict[config["model"]["model_name"]]

    # Initialize model
    model = ModelClass(config["model"]["num_classes"])

    lightning_module = HandwritingClassifier(model, config)

    # Init trainer
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["max_epochs"],
        logger=comet_logger,
        log_every_n_steps=config["trainer"]["log_every_n_steps"],
    )

    # Fit model
    trainer.fit(lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    main(config)
