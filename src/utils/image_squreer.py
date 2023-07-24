from PIL import Image
import numpy as np

def split_and_stack_image_to_square(img_path):
    # Open the image
    img = Image.open(img_path)
    width, height = img.size

    # Calculate the necessary number of splits and the new width of each split
    num_splits = int(np.ceil(width / np.sqrt(height * width)))
    new_width = height * num_splits

    # If the new width is larger than the original width, trim the image
    if new_width > width:
        left = (width - new_width) / 2
        right = (width + new_width) / 2
        img = img.crop((left, 0, right, height))
        width, height = img.size

    # Calculate the final width of each split
    split_width = width // num_splits

    # Split the image and stack the parts vertically
    parts = [img.crop((i * split_width, 0, (i + 1) * split_width, height)) for i in range(num_splits)]
    stacked_img = Image.new('L', (split_width, num_splits * height))

    for i, part in enumerate(parts):
        stacked_img.paste(part, (0, i * height))

    return stacked_img

# Path to your image
img_path = r'C:\Users\kfirs\PycharmProjects\FinalProject\data\data_sets\Development\first_set\train\1\ImagesMedianBW_row_1.jpg'

# Split and stack the image
stacked_img = split_and_stack_image_to_square(img_path)

# Save the final image
stacked_img.save(r'C:\Users\kfirs\PycharmProjects\FinalProject\data\data_sets\Development\first_set\train\1\ImagesMedianBW_row_square_1.jpg')
