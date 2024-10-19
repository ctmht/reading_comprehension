from src.data.hugging_face_data_loader import HuggingFaceDataLoader
from src.data.task_creator import CodeAnalysisTasks


def main():
    data_loader = HuggingFaceDataLoader("data/")
    dd = data_loader.load_chunked_files()

    df = dd.compute()  # Convert Dask to pandas DataFrame

    task_generator = CodeAnalysisTasks(
        df, max_rows=1
    )  # Process only the first 1000 rows
    task_generator.generate_tasks()
    task_generator.save_tasks()


if __name__ == "__main__":
    main()
