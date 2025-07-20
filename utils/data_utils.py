import os
import requests
import zipfile

def download_dataset(dataset_path, url):
    response = requests.get(url)
    with open("dataset.zip", "wb") as ds_zip:
        if response.ok:
            ds_zip.write(response.content)
        else:
            raise RuntimeError("Dataset does not exist and URL is inaccessible!")
    with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
        zip_ref.extractall(dataset_path)
    os.remove("dataset.zip")
    print(f"{url} downloaded!")