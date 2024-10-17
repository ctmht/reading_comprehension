from src.data.hugging_face_data_loader import HuggingFaceDataLoader
from src.models.llama_7b import Llama7B

RAW_TEXT_DATASET = (
    "hf://datasets/Nan-Do/code-search-net-python/data/train-*-of-*.parquet"
)


def main() -> None:
    data_loader = HuggingFaceDataLoader("data/")

    # Step 1: Pull the dataset, chunk it, and save it
    # data_loader.pull_and_chunk_dataset(RAW_TEXT_DATASET)

    # Step 2: Load the chunked data files
    chunked_data = data_loader.load_chunked_files()

    # Initialize the model with the loaded data
    model = Llama7B(data_loader, chunked_data)
    model.finetune()


if __name__ == "__main__":
    main()
