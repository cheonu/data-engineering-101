import kagglehub
import os


class DatasetDownload:
    def __init__(self, output_dir: str = "./models") -> None:
        self.output_dir = output_dir

    def download_kaggle_dataset(self, dataset_name: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        path = kagglehub.dataset_download(dataset_name, output_dir=self.output_dir)
        print("Path to dataset files:", path)
        return path

if __name__ == "__main__":
    downloader = DatasetDownload()
    downloader.download_kaggle_dataset("yasserh/titanic-dataset")