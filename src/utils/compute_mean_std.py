import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import yaml

def compute_mean_std_grayscale(data_loader):
    # Variable to accumulate sum and squared sum for grayscale images
    pixel_sum = torch.zeros(1)
    pixel_squared_sum = torch.zeros(1)
    total_pixels = 0

    # Iterate over all images in the dataset
    for images, _ in tqdm(data_loader, desc="Computing mean and std"):
        # Reshape images from (B, C, H, W) to (B*H*W, C)
        reshaped_images = images.view(-1, 1)

        # Update sum and squared sum
        pixel_sum += reshaped_images.sum(dim=0)
        pixel_squared_sum += (reshaped_images ** 2).sum(dim=0)

        # Update total pixel count
        total_pixels += reshaped_images.size(0)

    # Compute mean and std
    mean = pixel_sum / total_pixels
    std = (pixel_squared_sum / total_pixels - mean**2)**0.5

    return mean, std

def update_yaml_with_mean_std(config_file_path, mean, std):
    # Load the existing config
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the 'data' key with mean and std values
    config['data']['mean'] = mean.tolist()[0]
    config['data']['std'] = std.tolist()[0]

    # Write the updated config back to the file
    with open(config_file_path, 'w') as file:
        yaml.safe_dump(config, file)


if __name__ == '__main__':

    # Define dataset and data loader
    dataset_path = "/homes/kfirs/PycharmProjects/FinalProject/data/data_sets/Development/first_set_204_text_detection_filtering/train"  # Replace with the path to your dataset
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.Resize((128, 128)),transforms.ToTensor()])
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Compute mean and std for the dataset
    mean, std = compute_mean_std_grayscale(data_loader)

    # Update the config.yaml file with the computed mean and std
    config_file_path = "/homes/kfirs/PycharmProjects/FinalProject/config.yaml"  # Replace with the path to your config.yaml file
    update_yaml_with_mean_std(config_file_path, mean, std)
