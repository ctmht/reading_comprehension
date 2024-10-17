import os

import dask.dataframe as dd
import pandas as pd
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
        self.raw_dataset = None
        self.dataset = None

    def pull_and_chunk_dataset(self, dataset_link: str, chunk_size: int = 5000000):
        """
        Pull a dataset from Hugging Face, split it into smaller chunks, and save to the output directory.

        Args:
            dataset_link (str): The dataset link for Hugging Face dataset.
            chunk_size (int): Number of rows per chunk. Adjust this size based on the file size limitations.

        Raises:
            Exception: If there's an error pulling or saving the dataset.
        """
        try:
            # Load the dataset from Hugging Face
            self.raw_dataset = load_dataset("Nan-Do/code-search-net-python")

            # Convert the dataset to a pandas DataFrame
            df = self.raw_dataset.to_pandas()

            # Calculate the number of chunks based on the desired chunk size
            num_chunks = (len(df) // chunk_size) + 1

            # Ensure the output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Split and save each chunk as a separate Parquet file
            for i in range(num_chunks):
                chunk_df = df[i * chunk_size : (i + 1) * chunk_size]
                chunk_path = os.path.join(self.output_dir, f"data_chunk_{i}.parquet")
                chunk_df.to_parquet(chunk_path)
                print(f"Saved chunk {i+1}/{num_chunks} to {chunk_path}")

        except Exception as e:
            print(f"Error pulling or saving the dataset: {e}")
            raise

    def load_chunked_files(self) -> dd.DataFrame:
        """
        Load chunked dataset files from the output directory using Dask.

        Returns:
            dd.DataFrame: Dask dataframe of the loaded dataset chunks.

        Raises:
            ValueError: If there are no chunked files in the output directory.
        """
        try:
            # Use Dask to load all Parquet chunk files in the output directory
            chunk_files = os.path.join("data/original_data/", "data_chunk_*.parquet")
            dask_df = dd.read_parquet(chunk_files)

            print(f"Loaded chunked files from {self.output_dir}")
            return dask_df
        except Exception as e:
            print(f"Error loading chunked files: {e}")
            raise

    def load_local_file(self, file_name: str, chunk_size: int = 100000):
        """
        Load a dataset from a local file.
        This method attempts to load the dataset from a local file and sets it to self.raw_dataset.

        Args:
            file_name (str): The name of the file to load.
            chunk_size (int): Not used with Dask.

        Raises:
            Exception: If there's an error loading the dataset.
        """
        try:
            file_path = os.path.join(self.output_dir, file_name)

            if file_name.endswith(".csv"):
                self.raw_dataset = dd.read_csv(file_path)
            elif file_name.endswith(".parquet"):
                self.raw_dataset = dd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_name}")

            print("Local dataset loaded successfully with Dask")
            print(f"Dataset structure: {self.raw_dataset}")
            self._combine_dataset()
        except Exception as e:
            print(f"Error loading local dataset: {e}")
            raise

    def _combine_dataset(self):
        """
        Combine multiple partitions of a dataset into a single dataset.

        This method is called internally to merge multiple partitions.

        """
        # In Dask, combining partitions is already handled automatically, but if you want to
        # trigger a compute or additional processing, you can do so here.
        self.dataset = self.raw_dataset  # No explicit combination needed for Dask

    def print_dataset(self):
        """
        Print the first few rows of the dataset.

        This method is useful for quickly inspecting the loaded dataset.
        """
        print(self.dataset.head())
