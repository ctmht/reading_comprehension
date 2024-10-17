import csv
import os

import dask.dataframe as dd
from datasets import Dataset, DatasetDict, load_dataset


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

    def load_dataset(self, link: str):
        """
        Load a dataset from Hugging Face using the provided link.

        This method attempts to load the dataset and combines multiple partitions if present.

        Args:
            link (str): The Hugging Face dataset link.

        Raises:
            Exception: If there's an error loading the dataset.
        """
        try:
            self.raw_dataset = load_dataset(link)
            print("Dataset loaded successfully")
            print(f"Dataset structure: {self.raw_dataset}")
            self._combine_dataset()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def _combine_dataset(self):
        """
        Combine multiple partitions of a dataset into a single dataset.

        This method is called internally by load_dataset to merge multiple partitions.

        Args:
            ds (dict): A dictionary containing dataset partitions.
        """
        self.dataset = self.raw_dataset[next(iter(self.raw_dataset))]
        for partition in list(self.raw_dataset.keys())[1:]:
            self.dataset = self.dataset.concatenate_datasets(
                [self.raw_dataset[partition]]
            )

    def output_dataset(self, split: str = None) -> DatasetDict:
        """
        Return the raw dataset as a DatasetDict object, or a specific split as a Dataset object.

        Args:
            split (str, optional): The name of the split to return. If None, returns the entire DatasetDict.

        Returns:
            DatasetDict | Dataset: The entire DatasetDict if no split is specified, or a Dataset object for a specific split.

        Raises:
            ValueError: If the dataset has not been loaded yet or if the specified split doesn't exist.
        """
        if self.raw_dataset is None:
            raise ValueError(
                "Dataset has not been loaded. Please load a dataset first."
            )

        if split is None:
            return self.raw_dataset
        elif split in self.raw_dataset:
            return self.raw_dataset[split]
        else:
            available_splits = list(self.raw_dataset.keys())
            raise ValueError(
                f"Split '{split}' not found. Available splits are: {available_splits}"
            )

    def load_local_file(self, file_name: str, chunk_size: int = 100000):
        """
        Load a dataset from a local file.
        This method attempts to load the dataset from a local file and sets it to self.raw_dataset.

        Args:
            file_name (str): The name of the file to load.
            chunk_size (int): The number of rows to load at a time for Parquet files.

        Raises:
            Exception: If there's an error loading the dataset.
        """
        try:
            _, extension = os.path.splitext(file_name)
            extension = extension.lower()
            file_path = os.path.join(self.output_dir, file_name)

            if extension == ".csv":
                df = dd.read_csv(file_path)
                self.raw_dataset = Dataset.from_pandas(
                    df.compute()
                )  # Convert Dask DF to pandas before loading to Dataset
            elif extension == ".parquet":
                print("Parquet file detected")
                # Use dask to read Parquet files
                df = dd.read_parquet(file_path)
                self.raw_dataset = Dataset.from_pandas(
                    df.compute()
                )  # Convert Dask DF to pandas before loading to Dataset
            else:
                raise ValueError(f"Unsupported file type: {extension}")

            print("Local dataset loaded successfully")
            print(f"Dataset structure: {self.raw_dataset}")
            self._combine_dataset()
        except Exception as e:
            print(f"Error loading local dataset: {e}")
            raise

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
            dask_df = dd.from_pandas(self.dataset.to_pandas(), npartitions=10)
            dask_df.to_parquet(output_path_parquet)
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
            dask_df = dd.from_pandas(self.dataset.to_pandas(), npartitions=10)
            dask_df.to_csv(output_path_csv, single_file=True)
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
