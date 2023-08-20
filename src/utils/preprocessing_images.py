import cv2
import numpy as np
import os
from pathlib import Path
from scipy.io import loadmat


def load_data(image_file, mat_file):
    # Load mat file
    data = loadmat(mat_file)
    data['peaks_indices'] = (data['peaks_indices'].flatten() * data['SCALE_FACTOR'].flatten()[0]).astype(int)

    # Load image
    img = cv2.imread(image_file, 0)  # Load in grayscale
    return img, data


def detect_edges_in_image_with_border_exclude(img_np, threshold=220, border_exclude=700):
    """
    Detect edges in the middle row of an image, excluding pixels from the borders.

    Parameters:
    - img_np: numpy ndarray of the image (grayscale)
    - threshold: Color difference to consider an edge
    - border_exclude: Number of pixels to exclude from both left and right borders

    Returns:
    - int: Number of edges detected
    """

    # Extract the first third of
    middle_row = 2*(img_np.shape[0] // 3)
    pixels = img_np[middle_row, border_exclude:-border_exclude]

    # Compute the difference between adjacent pixels and count edges
    edge_count = 0
    for i in range(1, len(pixels)):
        if abs(pixels[i] - pixels[i - 1]) > threshold:
            edge_count += 1

    return edge_count


def contains_text(img, edge_threshold=50, color_difference_threshold=180):
    """
    Check if the image contains text based on the number of detected edges.

    Parameters:
    - img: PIL Image object
    - edge_threshold: Number of edges that indicates the presence of text
    - color_difference_threshold: Color difference to consider an edge

    Returns:
    - bool: True if the image likely contains text, False otherwise
    """
    # Count the number of edges in the image
    num_edges = detect_edges_in_image_with_border_exclude(img, color_difference_threshold)

    # Check if the number of edges exceeds the given threshold
    return num_edges > edge_threshold

def is_mostly_white(img, row_threshold=0.87, p_t=0.97, border_exclude=700):
    """
    Check if the image is mostly white.
    """
    white_rows = 0
    img = img[:, border_exclude:-border_exclude]
    for row in img:
        if np.mean(row) / 255 >= p_t:  # if row is mostly white
            white_rows += 1
    return (white_rows / img.shape[0]) >= row_threshold


def split_rows(img, train_ratio, peaks_indices, test_indices, p_t):
    data = []
    mean_row_height = np.mean(np.diff(peaks_indices)).astype(int)

    # iterate the rows in the image by the data['peaks_indices'] values
    for i in range(len(peaks_indices) - 1):
        row = img[peaks_indices[i]:(peaks_indices[i]+mean_row_height)]

        # check if peaks_indices is between test_indices
        if (peaks_indices[i]+mean_row_height) >= test_indices[0] and (peaks_indices[i]+mean_row_height) <= test_indices[1]:
            print(f'row {i} is between test_indices')
            continue

        # skip if row is not contain text
        if not contains_text(row) or is_mostly_white(row, p_t):
            print(f'row {i} not conatain text')
        else:
            data.append(row)

    # Calculate split indices
    total_rows = len(data)
    train_end = int(total_rows * train_ratio)

    # Split data to train, val and test
    test_row = [img[test_indices[0]:test_indices[1]]]

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


def split_image_to_sets(image_file: str, mat_file: str, train_ratio: float, output_folder: str, p_t: float):
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
    train_rows, val_rows, test_row = split_rows(img, train_ratio, peaks_indices, test_indices, p_t)

    save_rows(train_rows, output_folder, 'train', label, source_type)
    save_rows(val_rows, output_folder, 'val', label, source_type)
    save_rows(test_row, output_folder, 'test', label, source_type)

    return train_rows, val_rows, test_row


# Usage:
if __name__ == '__main__':
    K = 100
    ratio = 0.8
    dev_raw_data_paths_dict = {
        'data_rotated_path': {
            'path': r'/homes/kfirs/PycharmProjects/FinalProject/data/raw_data/Testing/1_ImagesRotated',
            'pixel_threshold': 0.96},
        'data_median_bw_path': {
            'path': r'/homes/kfirs/PycharmProjects/FinalProject/data/raw_data/Testing/2_ImagesMedianBW',
            'pixel_threshold': 0.96},
        'data_lines_removed_median_bw_path': {
            'path': r'/homes/kfirs/PycharmProjects/FinalProject/data/raw_data/Testing/3_ImagesLinesRemovedBW',
            'pixel_threshold': 0.97},
        'data_lines_removed_path': {
            'path': r'/homes/kfirs/PycharmProjects/FinalProject/data/raw_data/Testing/4_ImagesLinesRemoved',
            'pixel_threshold': 0.97}
    }
    data_dark_lines_path = r'/homes/kfirs/PycharmProjects/FinalProject/data/raw_data/Testing/5_DataDarkLines'


    output_folder = fr'/homes/kfirs/PycharmProjects/FinalProject/data/data_sets/Testing/Testing_set_{K}_ratio_{ratio}'

    # iterate over the dev_raw_data_paths_dict
    for key, value in dev_raw_data_paths_dict.items():
        raw_data_path = value['path']
        pixel_threshold = value['pixel_threshold']
        # iterate over the raw data path

        for index, (image_file, dark_lines_mat) in enumerate(zip(Path(raw_data_path).glob('*.jpg'),
                                                                 Path(data_dark_lines_path).glob('*.mat'))):
            if index == K:
                break
            split_image_to_sets(str(image_file), str(dark_lines_mat), ratio, output_folder, p_t=pixel_threshold)
