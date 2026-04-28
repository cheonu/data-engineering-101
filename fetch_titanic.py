import kagglehub


class dataset_download:
    def download_kaggle_dataset(dataset_name):
        path = kagglehub.dataset_download(dataset_name, output_dir='./models')
        print("Path to dataset files:", path)

dataset_download.download_kaggle_dataset("yasserh/titanic-dataset")