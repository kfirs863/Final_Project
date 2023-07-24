from PIL import Image
import numpy as np

class SplitAndStackImageToSquare(object):
    def __call__(self, img):
        # Your function code here:
        width, height = img.size
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
