import yaml
from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import pytorch_lightning as pl
from src.lightning_module import HandwritingClassifier
from src.models import CNNModel
from src.utils.custom_transformer import SplitAndStackImageToSquare


def main(test_config):

    # Set image size
    IMAGE_SIZE = test_config["data"]["image_size"]

    # Common preprocessing transformations
    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        SplitAndStackImageToSquare(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    # datasets
    test_dataset = ImageFolder(Path(test_config["data"]["test_data_path"]), transform=test_transforms)

    # Initialize model
    model = CNNModel(num_classes=test_config["model"]["num_classes"])

    # Path to your checkpoint
    checkpoint_path = Path('checkpoints/development_checkpoint/best-epoch=137-val_accuracy=0.85.ckpt')

    # Initialize lightning module
    lightning_module = HandwritingClassifier.load_from_checkpoint(checkpoint_path, model=model, config=test_config,test_dataset=test_dataset, map_location='cpu')

    # Evaluate using trainer
    trainer = pl.Trainer(accelerator='cpu')

    # Evaluate the model on the test set
    trainer.test(lightning_module)

if __name__ == '__main__':

    # Load config file
    with open("develop_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    main(config)