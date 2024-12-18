import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import kagglehub
import yaml

# Download latest version
# path = kagglehub.dataset_download("uom190346a/water-quality-and-potability")

# print("Path to dataset files:", path)

# for file_name in os.listdir(path):
#     if file_name.endswith(".csv"):  # Check if the file is a CSV
#         csv_path = os.path.join(path, file_name)
#         print("Reading CSV file:", csv_path)
#         data = pd.read_csv(csv_path)  # Read the CSV file into a pandas DataFrame

def download_dataset(dataset_name):
    try:
        return kagglehub.dataset_download(dataset_name)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset '{dataset_name}': {e}")

def get_csv_files(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path '{path}' does not exist.")
        return [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(".csv")]
    except Exception as e:
        raise RuntimeError(f"Error retrieving CSV files from '{path}': {e}")

def read_csv_file(csv_path):
    try:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"File '{csv_path}' does not exist.")
        return pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file '{csv_path}': {e}")
    
def load_params(filepath: str) -> float:
    try:
        with open(filepath,"r") as file:
            params = yaml.safe_load(file)
        return params["data_collection"]["test_size"]
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}:{e}")

# test_size = yaml.safe_load(open("params.yaml"))["data_collection"]["test_size"]

def split_data(data: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    try:
        return train_test_split(data, test_size=test_size, random_state=42)
    except ValueError as e:
        raise ValueError(f"Error Splitting data : {e}")

# train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

def save_data(df: pd.DataFrame, filepath: str) -> None:
    try:
        df.to_csv(filepath, index = False)
    except Exception as e:
        raise Exception(f"Error saving data to {filepath}: {e}")

# data_path = os.path.join("data", "raw")

# os.makedirs(data_path)

# train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)


def main():
    dataset_name = "uom190346a/water-quality-and-potability"
    params_filepath = "params.yaml"
    raw_data_path = os.path.join("data", "raw")
    try:
        path = download_dataset(dataset_name)
        print("Path to dataset files:", path)

        csv_files = get_csv_files(path)
        if not csv_files:
            print("No CSV files found in the dataset directory.")
            return

        for csv_path in csv_files:
            print("Reading CSV file:", csv_path)
            data = read_csv_file(csv_path)
            print(data.head())
        
        test_size = load_params(params_filepath)
        train_data, test_data = split_data(data, test_size)
        os.makedirs(raw_data_path)

        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()