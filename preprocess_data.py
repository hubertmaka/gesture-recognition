import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib as matplb


def print_libs_version() -> None:
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {keras.__version__}")
    print(f"Pandas Version: {pd.__version__}")
    print(f"Numpy Version: {np.__version__}")
    print(f"Matplotlib Version: {matplb.__version__}")
    print(f"Python Version: {sys.version}")


def get_sequence_len(path: str) -> int:
    return len([file for file in os.listdir(path) if file.endswith(".jpg")])


def get_directory_path(catalog_id: int, main_directory: str, batch_size: int) -> str:
    batch_num = (catalog_id - 1) // batch_size + 1
    batch_dir = f"kat{batch_num}"
    directory_path = os.path.join(main_directory, batch_dir, str(catalog_id))
    return directory_path


def preprocess_dataset() -> None:
    jester_dataset_dir_path = os.path.join("/", "content", "drive", "MyDrive", "gesture-recognition")
    train_info_path = os.path.join(jester_dataset_dir_path, "info", "jester-v1-train.csv")
    val_info_path = os.path.join(jester_dataset_dir_path, "info", "jester-v1-validation.csv")
    test_info_path = os.path.join(jester_dataset_dir_path, "info", "jester-v1-test.csv")
    labels_info_path = os.path.join(jester_dataset_dir_path, "info", 'jester-v1-labels.csv')
    video_dir_path = os.path.join(jester_dataset_dir_path, "data")

    pd.options.mode.copy_on_write = True
    dir_split_batch_size = 9000

    column_names = ["video_id", "label"]
    train_df = pd.read_csv(train_info_path, names=column_names, header=None, delimiter=";")
    val_df = pd.read_csv(val_info_path, names=column_names, header=None, delimiter=";")
    test_df = pd.read_csv(test_info_path, names=column_names, header=None, delimiter=";")

    jester_labels = {}
    with open(labels_info_path) as f:
        for idx2, line in enumerate(f):
            jester_labels[line.strip()] = idx2

    train_df["label_id"] = train_df["label"].apply(lambda label: jester_labels.get(label))
    val_df["label_id"] = val_df["label"].apply(lambda label: jester_labels.get(label))

    train_df["path"] = train_df["video_id"].apply(
        lambda video_id: get_directory_path(
            main_directory=video_dir_path,
            catalog_id=video_id,
            batch_size=dir_split_batch_size
        )
    )
    val_df["path"] = val_df["video_id"].apply(
        lambda video_id: get_directory_path(
            main_directory=video_dir_path,
            catalog_id=video_id,
            batch_size=dir_split_batch_size
        )
    )

    train_df["seq_len"] = train_df["path"].apply(lambda path: get_sequence_len(path))
    val_df["seq_len"] = val_df["path"].apply(lambda path: get_sequence_len(path))

    train_df.to_csv(os.path.join(jester_dataset_dir_path, "info", "train_df.csv"), index=False)
    val_df.to_csv(os.path.join(jester_dataset_dir_path, "info", "val_df.csv"), index=False)


def main(verbose: bool = False) -> None:
    if verbose:
        print_libs_version()
    preprocess_dataset()


if __name__ == "__main__":
    main(verbose=True)

