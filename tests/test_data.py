import os
import pytest
import torch
import json
from google.cloud import storage

BUCKET_NAME = "movies-mlops-clean"
FILE_NAMES = ["train.pt", "test.pt", "val.pt"]
PATH_TO_DOWNLOAD = "tests/temp_data/"

# save locally
train_data_path = "tests/temp_data/train.pt" 


@pytest.fixture
def train_dataset():
    # download & load data
    storage_client = storage.Client(project="zeroshots")

    if not os.path.exists(PATH_TO_DOWNLOAD):
        os.mkdir(PATH_TO_DOWNLOAD)

    print("Downloading data...")
    bucket = storage_client.bucket(BUCKET_NAME)

    for file in FILE_NAMES:
        blob = bucket.blob(file)
        a = blob.download_to_filename(PATH_TO_DOWNLOAD + file)
        print(a)
        
    train_dataset = torch.load(train_data_path)

    return train_dataset


def test_if_bucket_exists():
    storage_client = storage.Client(project="zeroshots")
    assert storage_client.get_bucket(BUCKET_NAME)


def test_loaded_data_not_empty(train_dataset):
    assert len(train_dataset) != 0


def test_labels(train_dataset):
    # assert that all labels are represented
    assert len(train_dataset['user']) == 1
    assert len(train_dataset['movie']) == 1
    assert len(train_dataset['user', 'rates', 'movie']) == 3
    assert len(train_dataset['movie', 'rev_rates', 'user']) == 1


def test_datapoint_size(train_dataset):
    # assert size of each datapoint
    assert train_dataset['user'].x.shape == torch.Size([671, 671])
    assert train_dataset['movie'].x.shape == torch.Size([9025, 404])
    assert train_dataset['user', 'rates', 'movie'].edge_index.shape == torch.Size([2, 79848])
    assert train_dataset['user', 'rates', 'movie'].edge_label.shape == torch.Size([79848])
    assert train_dataset['user', 'rates', 'movie'].edge_label_index.shape == torch.Size([2, 79848])
    assert train_dataset['movie', 'rev_rates', 'user'].edge_index.shape == torch.Size([2, 79848])
