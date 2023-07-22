from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat
import os


def load_data(image_file, mat_file):
    # Load mat file
    data = loadmat(mat_file)
    data['peaks_indices'] = (data['peaks_indices'].flatten() * data['SCALE_FACTOR'].flatten()[0]).astype(int)

    # Load image
    img = cv2.imread(image_file, 0)  # Load in grayscale
    return img, data


def is_mostly_white(img, row_threshold=0.90, pixel_threshold=0.95):
    white_rows = 0
    for row in img:
        if np.mean(row) / 255 >= pixel_threshold:  # if row is mostly white
            white_rows += 1
    return (white_rows / img.shape[0]) >= row_threshold


def split_rows(img, train_ratio, peaks_indices, test_indices):
    data = []

    # iterate the rows in the image by the data['peaks_indices'] values
    for i in range(len(peaks_indices) - 1):
        row = img[peaks_indices[i]:peaks_indices[i + 1]]

        # skip if peak_indices are equal to test_indices
        if peaks_indices[i] == test_indices[0] or peaks_indices[i] == test_indices[1]:
            continue

        # check row image contain some text
        if is_mostly_white(row):
            print(f'row {i} is mostly white')
            # display image
            # cv2.imshow('image', row)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            data.append(row)

    # Calculate split indices
    total_rows = len(data)
    train_end = int(total_rows * train_ratio)

    # Split data to train, val and test
    test_row = [img[test_indices[0]:test_indices[1]]]

    # shuffle data
    np.random.shuffle(data)

    train_rows = data[:train_end]
    val_rows = data[train_end:]

    return train_rows, val_rows, test_row


def resize_rows(rows, size):
    resized_rows = []
    for row in rows:
        resized_rows.append(cv2.resize(row, size))
    return resized_rows


def save_rows(rows, folder, prefix, label, source_type):
    # Save each row as a separate image file
    for i, row in enumerate(rows):
        destination_folder = Path(folder, prefix, label)
        if not destination_folder.exists():
            os.makedirs(destination_folder, exist_ok=True)
        cv2.imwrite(os.path.join(folder, prefix, label, f'{source_type}_row_{i}.jpg'), row)


def split_image_to_sets(image_file: str, mat_file: str, train_ratio: float, output_folder: str):
    # use Pathlib to assert that the image and mat files exist
    assert Path(image_file).exists(), f'Image file {image_file} does not exist'
    assert Path(mat_file).exists(), f'Mat file {mat_file} does not exist'

    # assert that the image suffix number is equal to the mat file suffix number
    image_suffix = image_file.split('_')[-1].split('.')[0]
    mat_suffix = mat_file.split('_')[-1].split('.')[0]
    assert int(image_suffix) == int(
        mat_suffix), f'Image file {image_file} and mat file {mat_file} suffix numbers do not match'

    # check if output folder exists, if not create it
    if not Path(output_folder).exists():
        os.makedirs(output_folder)

    label = str(int(image_suffix))
    source_type = Path(image_file).parent.name

    img, data = load_data(image_file, mat_file)
    peaks_indices = data['peaks_indices']
    test_indices = (data['top_test_area'].flatten()[0], data['bottom_test_area'].flatten()[0])
    train_rows, val_rows, test_row = split_rows(img, train_ratio, peaks_indices, test_indices)

    save_rows(train_rows, output_folder, 'train', label, source_type)
    save_rows(val_rows, output_folder, 'val', label, source_type)
    save_rows(test_row, output_folder, 'test', label, source_type)

    return train_rows, val_rows, test_row


# Usage:
if __name__ == '__main__':
    data_dark_lines_path = r'C:\Users\kfirs\PycharmProjects\FinalProject\data\raw_data\Development\DataDarkLines'
    data_median_bw_path = r'C:\Users\kfirs\PycharmProjects\FinalProject\data\raw_data\Development\ImagesMedianBW'
    for image_file, dark_lines_mat in zip(Path(data_median_bw_path).glob('*.jpg'),
                                          Path(data_dark_lines_path).glob('*.mat')):
        split_image_to_sets(str(image_file), str(dark_lines_mat), 0.8,
                            r'C:\Users\kfirs\PycharmProjects\FinalProject\data\data_sets\Development\first_set')
