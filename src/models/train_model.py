
import logging
import os
import sys
import hydra
from src.models import _PATH_DATA
# import importlib.util


import pandas as pd
from arango import ArangoClient
from tqdm import tqdm
import numpy as np
import itertools
import requests
import sys
from arango import ArangoClient

import torch
import torch.nn.functional as F
# from torch.nn import Linear
# from arango import ArangoClient
# import torch_geometric.transforms as T
# from torch_geometric.nn import SAGEConv, to_hetero
# from torch_geometric.transforms import RandomLinkSplit, ToUndirected
# from sentence_transformers import SentenceTransformer
# from torch_geometric.data import HeteroData
import yaml



# from google.cloud import storage  # type: ignore
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
from src.models.model import Model


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


# def test(data: torch_geometric.data.Data, model: torch.nn.Module) -> float:
def test(data, model):
    """
    Evaluates model on data and returns accuracy.
    :param model: Model to be evaluated
    :param data: Data to evaluate on
    :return: accuracy
    """


    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'movie'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['user', 'movie'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


@hydra.main(config_path="../config", config_name="default_config.yaml")
def train(config: DictConfig) -> None:    
    """
    Trains the model with hyperparameters in config on train data,
    saves the model and evaluates it on test data.
    :param config: Config file used for Hydra
    :return:
    """
    # log = logging.getLogger(__name__)
    # print = log.info
    # # wandb.init(project="Project-MLOps", entity="Project-MLOps")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training")
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiments.hyperparams
    # wandb.config = hparams
    torch.manual_seed(hparams["seed"])
    orig_cwd = hydra.utils.get_original_cwd()

    # Load data


    train_data = torch.load(os.path.join(_PATH_DATA, "train.pt"))
    val_data = torch.load(os.path.join(_PATH_DATA, "val.pt"))
    test_data = torch.load(os.path.join(_PATH_DATA, "test.pt"))


    # Model
    metadata = (['user', 'movie'],
            [('user', 'rates', 'movie'), ('movie', 'rev_rates', 'user')])

    model = Model(metadata,
        hidden_channels=hparams["hidden_channels"],
    )

    model = model.to(device)

    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=5e-4)
    




    # Train model
    for epoch in range(hparams["epochs"]):

        #TRAIN
        optimizer.zero_grad()
        pred = model(train_data.x_dict, train_data.edge_index_dict,
        train_data['user', 'movie'].edge_label_index)
        target = train_data['user', 'movie'].edge_label
        weight = torch.bincount(train_data['user', 'movie'].edge_label)
        weight = weight.max() / weight
        loss = weighted_mse_loss(pred, target, weight)
        loss.backward()
        optimizer.step()

        
        # Compute rmse
        train_rmse = test(train_data, model)
        val_rmse = test(val_data, model)
        test_rmse = test(test_data, model)

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
 #       wandb.log({"Training loss": loss})




        # # Clear gradients
        # optimizer.zero_grad()
        # # Perform a single forward pass
        # out = model(train_data.x_dict, train_data.edge_index_dict,
        #          train_data['user', 'movie'].edge_label_index)
        # target = train_data['user', 'movie'].edge_label
        # # Compute the loss solely based on the training nodes


        # loss = weighted_mse_loss(out, target, weight)
        # # Derive gradients
        # loss.backward()
        # # Update parameters based on gradients
        # optimizer.step()
        # # Append results
        # train_loss.append(loss.item())

        # # Compute rmse
        # train_rmse = test(train_data)
        # val_rmse = test(val_data)
        # test_rmse = test(test_data)

#         # print
#         print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
#           f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
#  #       wandb.log({"Training loss": loss})



    # Save model
    directory = orig_cwd + "/models/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + hparams["checkpoint_name"]
    checkpoint = {'hidden_channels': hparams["hidden_channels"],
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filename)    



if __name__ == "__main__":
    train()    