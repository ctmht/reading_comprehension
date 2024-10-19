from src.data.hugging_face_data_loader import HuggingFaceDataLoader
from src.models.llama_7b import Llama7B
import os
import torch

def get_latest_epoch():
    epochs = [int(d.split('_')[-1]) for d in os.listdir('.') if d.startswith('finetuned_model_epoch_')]
    return max(epochs) if epochs else 0

RAW_TEXT_DATASET = (
    "hf://datasets/Nan-Do/code-search-net-python/data/train-*-of-*.parquet"
)


def main() -> None:
    print(torch.cuda.is_available())  # Should return True
    print(torch.version.cuda)  # Should be 11.x
    data_loader = HuggingFaceDataLoader("data/")

    # Step 1: Pull the dataset, chunk it, and save it
    # data_loader.pull_and_chunk_dataset(RAW_TEXT_DATASET)

    # Step 2: Load the chunked data files
    chunked_data = data_loader.load_chunked_files()

    # Initialize the model with the loaded data
    model = Llama7B(data_loader, chunked_data)

    # Print dataset information
    model.print_dataset_info()
    start_epoch = get_latest_epoch()
    # Fine-tune the model
    model.finetune(start_epoch=start_epoch, num_epochs=3)


if __name__ == "__main__":
    main()
