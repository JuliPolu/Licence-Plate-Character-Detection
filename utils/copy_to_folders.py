import shutil
from pathlib import Path

def copy_to_folders(
    dest_folder: str,
    source_folder: str,
    dataset_modes: list,
    subsets_dict: dict,
) -> None:
    dest_folder = Path(dest_folder)
    source_folder_all = Path(source_folder)

    for mode in dataset_modes:
        if mode == 'valid':
            mode_source_folder = source_folder_all / 'train'
        else:
            mode_source_folder = source_folder_all / mode
        
        dest_path_image = dest_folder / 'images' / mode
        dest_path_labels = dest_folder / 'labels' / mode

        dest_path_image.mkdir(parents=True, exist_ok=True)
        dest_path_labels.mkdir(parents=True, exist_ok=True)

        for image in subsets_dict[mode]:
            initial_path_image = mode_source_folder / image
            destination_path_image = dest_path_image / image
            shutil.copy(initial_path_image, destination_path_image)

            initial_path_labels = mode_source_folder / f"{image.split('.')[0]}.txt"
            destination_path_labels = dest_path_labels / f"{image.split('.')[0]}.txt"
            shutil.copy(initial_path_labels, destination_path_labels)
