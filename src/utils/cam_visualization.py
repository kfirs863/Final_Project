import torch
import yaml
from matplotlib import pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms

import cv2
import numpy as np

from src.lightning_module import HandwritingClassifier
from src.models import CNNModel
from src.utils.custom_transformer import SplitAndStackImageToSquare


class CAMModule(torch.nn.Module):
    def __init__(self, pl_module):
        super().__init__()
        self.pl_module = pl_module
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.pl_module.model.features(x)
        h = x.register_hook(self.activations_hook)
        x = self.pl_module.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.pl_module.model.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.pl_module.model.features(x)


def visualize_cam(image, cam_model, IMAGE_SIZE):
    # Forward pass
    output = cam_model(image)
    _, predicted = torch.max(output, 1)
    output[0][predicted].backward()
    gradients = cam_model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = cam_model.get_activations(image).detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.clamp(heatmap, min=0)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()  # Move to CPU and convert to numpy

    # Resize heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))

    # Get original image (assuming it's in [0, 1] range and single channel)
    original_image = image[0, 0].cpu().numpy()

    # Convert heatmap to colormap
    colormap = plt.get_cmap('jet')
    heatmap_colormap = (colormap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)

    # Convert original image to 3 channels
    original_image_3ch = np.stack([original_image] * 3, axis=2)

    # Convert original image to 3 channels and scale to [0, 255]
    original_image_3ch = (original_image * 255).astype(np.uint8)
    original_image_3ch = np.stack([original_image_3ch] * 3, axis=2)

    # Overlay heatmap on original image
    overlayed_image = cv2.addWeighted(original_image_3ch, 0.6, heatmap_colormap, 0.4, 0)

    # Show the image
    plt.imshow(overlayed_image)
    plt.axis('off')  # to remove axes
    plt.show()


def visualize_activations(image, lightning_module):
    model = lightning_module.model

    activations = []

    def hook_fn(module, input, output):
        if isinstance(module, torch.nn.Conv2d):
            activations.append(output)

    hooks = [layer.register_forward_hook(hook_fn) for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]

    with torch.no_grad():
        output = model(image)

    for hook in hooks:
        hook.remove()

    for i, activation in enumerate(activations):
        num_filters = activation.shape[1]
        n = int(np.ceil(np.sqrt(num_filters)))

        fig, axes = plt.subplots(n, n, figsize=(12, 12))

        for j in range(num_filters):
            ax = axes[j // n][j % n]
            act = activation[0, j].cpu().numpy()
            ax.imshow(act, cmap='viridis')
            ax.axis('off')

        plt.suptitle(f'Layer {i} Activations')
        plt.show()


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
        # transforms.RandAugment(),
        transforms.ToTensor(),
    ])

    val_dataset = ImageFolder(config["data"]["val_data_path"], transform=valid_and_test_transforms)
    train_dataset = ImageFolder(config["data"]["train_data_path"], transform=train_transforms)
    test_dataset = ImageFolder(config["data"]["test_data_path"], transform=valid_and_test_transforms)

    # Initialize model
    model = CNNModel(num_classes=config["model"]["num_classes"])

    # Path to your checkpoint
    checkpoint_path = '/tmp/pycharm_project_253/hand-writing-classification/8483ec259b8a4826aa9e02eaeec603cd/checkpoints/best-epoch=137-val_accuracy=0.85.ckpt'

    # Initialize lightning module
    lightning_module = HandwritingClassifier.load_from_checkpoint(checkpoint_path, model=model, config=config,
                                                                  train_dataset=train_dataset, val_dataset=val_dataset)

    # Initialize CAM module
    cam_model = CAMModule(lightning_module)

    # Retrieve a sample image from the test dataset
    selected_class = 199
    counter = 0
    for index, (image, class_number) in enumerate(test_dataset):
        if class_number != selected_class:
            continue

        # Add batch dimension
        image = image.unsqueeze(0) # Add batch dimension

        # Check if a GPU is available and if not, use the CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



        # Move the image to the same device as the model
        image = image.to(device) # Add this line

        # Visualize CAM for the specific image
        visualize_cam(image, cam_model, IMAGE_SIZE)

        # Visualize activations for the specific image
        visualize_activations(image, lightning_module)
        counter += 1
        if counter==5:
            break


if __name__ == '__main__':

    # Load config file
    with open("../../config.yaml", "r") as f:
        config = yaml.safe_load(f)

    main(config)
