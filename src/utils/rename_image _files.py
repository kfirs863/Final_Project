import os


def rename_files(path, file_extension='jpg'):
    # get list of files in folder
    files = os.listdir(path)

    # filter to keep only image files
    files = [f for f in files if f.endswith(file_extension)]

    # sort the files list
    files.sort()

    # iterate over files and rename them
    for i, file in enumerate(files,start=1):
        new_name = f'image_class_{i}.{file_extension}'
        old_file_path = os.path.join(path, file)
        new_file_path = os.path.join(path, new_name)

        os.rename(old_file_path, new_file_path)
        print(f'Renamed file {old_file_path} to {new_file_path}')

# usage example:
rename_files(r'C:\Users\kfirs\PycharmProjects\FinalProject\data\raw_data\1_ImagesRotated', 'jpg')
rename_files(r'C:\Users\kfirs\PycharmProjects\FinalProject\data\raw_data\2_ImagesMedianBW', 'jpg')
rename_files(r'C:\Users\kfirs\PycharmProjects\FinalProject\data\raw_data\3_ImagesLinesRemovedBW', 'jpg')
rename_files(r'C:\Users\kfirs\PycharmProjects\FinalProject\data\raw_data\4_ImagesLinesRemoved', 'jpg')
rename_files(r'C:\Users\kfirs\PycharmProjects\FinalProject\data\raw_data\5_DataDarkLines', 'mat')

