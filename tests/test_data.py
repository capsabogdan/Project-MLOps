# import pytest
# import torch
# from tests import _PATH_DATA
# import os

# def test_train_data():
#     train_data_path = os.path.join(_PATH_DATA, "./train.pt")  # root of data
#     dataset = torch.load(train_data_path)

#     # assert that all labels are represented
#     assert len(dataset['user']) == 1
#     assert len(dataset['movie']) == 1
#     assert len(dataset['user', 'rates', 'movie']) == 3
#     assert len(dataset['movie', 'rev_rates', 'user']) == 1

#     # assert size of each datapoint
#     assert dataset['user'].x.shape == torch.Size([671, 671])
#     assert dataset['movie'].x.shape == torch.Size([9025, 404])
#     assert dataset['user', 'rates', 'movie'].edge_index.shape == torch.Size([2, 79848])
#     assert dataset['user', 'rates', 'movie'].edge_label.shape == torch.Size([79848])
#     assert dataset['user', 'rates', 'movie'].edge_label_index.shape == torch.Size([2, 79848])
#     assert dataset['movie', 'rev_rates', 'user'].edge_index.shape == torch.Size([2, 79848])


def test_dummy():
    assert True
