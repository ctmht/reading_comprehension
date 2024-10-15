from src.data.hugging_face_data_loader import HuggingFaceDataLoader
from src.models.llama_7b import Llama7B

RAW_TEXT_DATASET = "Nan-Do/code-search-net-python"


def main() -> None:
    data_loader = HuggingFaceDataLoader("data/")
    data_loader.load_dataset(RAW_TEXT_DATASET)
    # data_loader.load_local_file("combined_data.parquet")

    model = Llama7B(data_loader)
    model.finetune()


if __name__ == "__main__":
    main()
