import os
import requests
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split


def download_file(url, save_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def extract_tar_gz(file_path, extract_path):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_path)


def remove_hidden_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("._"):
                os.remove(os.path.join(root, file))


def subsample_csv(input_path, output_path, fraction=0.01, random_state=42):
    try:
        df = pd.read_csv(input_path)
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="ISO-8859-1")
    subsample, _ = train_test_split(
        df, test_size=1 - fraction, random_state=random_state
    )
    subsample.to_csv(output_path, index=False)


def main():
    urls = [
        "https://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz",
        "https://snap.stanford.edu/decagon/bio-decagon-mono.tar.gz",
        "https://snap.stanford.edu/decagon/bio-decagon-ppi.tar.gz",
        "https://snap.stanford.edu/decagon/bio-decagon-targets.tar.gz",
    ]

    data_dir = "data"
    test_dir = os.path.join(data_dir, "test")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for url in urls:
        file_name = os.path.basename(url)
        tar_path = os.path.join(data_dir, file_name)

        print(f"Downloading {url}...")
        download_file(url, tar_path)

        print(f"Extracting {tar_path}...")
        extract_tar_gz(tar_path, data_dir)

        print(f"Deleting {tar_path}...")
        os.remove(tar_path)

    # Remove hidden files
    remove_hidden_files(data_dir)

    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        subsample_path = os.path.join(test_dir, csv_file)

        print(f"Creating subsample for {csv_file}...")
        subsample_csv(csv_path, subsample_path)

    print("Done!")


if __name__ == "__main__":
    main()
