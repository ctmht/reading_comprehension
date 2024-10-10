import csv
import os

import pyarrow.parquet as pq
from datasets import load_dataset


class HuggingFaceDataLoader:
    """
    A class for loading datasets from Hugging Face, combining them, and exporting to Parquet or CSV formats.

    This class provides functionality to load datasets from Hugging Face, combine multiple partitions
    if present, and export the resulting dataset to either Parquet or CSV format.

    Attributes:
        output_dir (str): The directory where output files will be saved.
        dataset: The loaded and combined dataset.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the HuggingFaceDataLoader with an output directory.

        Args:
            output_dir (str): The directory where output files will be saved.
        """
        self.output_dir = output_dir

    def load_database(self, link: str):
        """
        Load a dataset from Hugging Face using the provided link.

        This method attempts to load the dataset and combines multiple partitions if present.

        Args:
            link (str): The Hugging Face dataset link.

        Raises:
            Exception: If there's an error loading the dataset.
        """
        try:
            ds = load_dataset(link)
            print("Dataset loaded successfully")
            print(f"Dataset structure: {ds}")
            self._combine_dataset(ds)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def _combine_dataset(self, ds: dict):
        """
        Combine multiple partitions of a dataset into a single dataset.

        This method is called internally by load_dataset to merge multiple partitions.

        Args:
            ds (dict): A dictionary containing dataset partitions.
        """
        self.dataset = ds[next(iter(ds))]
        for partition in list(ds.keys())[1:]:
            self.dataset = self.dataset.concatenate_datasets([ds[partition]])

    def output_parquet(self, file_name: str):
        """
        Export the dataset to a Parquet file.

        Args:
            file_name (str): The name of the output file (without extension).

        Raises:
            Exception: If there's an error saving the Parquet file.
        """
        output_path_parquet = os.path.join(self.output_dir, file_name + ".parquet")

        try:
            self.dataset.to_parquet(output_path_parquet)
            print(f"Attempting to save data to: {output_path_parquet}")

            if os.path.exists(output_path_parquet):
                print(f"Parquet file successfully created at: {output_path_parquet}")
                print(
                    f"Parquet file size: {os.path.getsize(output_path_parquet)} bytes"
                )
            else:
                print(f"Error: Parquet file was not created at {output_path_parquet}")
        except Exception as e:
            print(f"Error saving to Parquet: {e}")
            raise

    def output_csv(self, file_name: str):
        """
        Export the dataset to a CSV file.

        Args:
            file_name (str): The name of the output file (without extension).

        Raises:
            Exception: If there's an error saving the CSV file.
        """
        output_path_csv = os.path.join(self.output_dir, file_name + ".csv")

        try:
            self.dataset.to_csv(output_path_csv)
            print(f"Attempting to save data to: {output_path_csv}")

            if os.path.exists(output_path_csv):
                print(f"CSV file successfully created at: {output_path_csv}")
                print(f"CSV file size: {os.path.getsize(output_path_csv)} bytes")
            else:
                print(f"Error: CSV file was not created at {output_path_csv}")
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            raise

    def print_dataset(self):
        """
        Print the first few rows of the dataset.

        This method is useful for quickly inspecting the loaded dataset.
        """
        print(self.dataset.head())
