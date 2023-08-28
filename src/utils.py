import datasets

def data_loader(file_path: str) -> datasets.Dataset:
    return datasets.load_from_disk(file_path)
