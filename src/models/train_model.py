
import logging
import os
import sys
import hydra
# import importlib.util


# from google.cloud import storage  # type: ignore
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_geometric  # type: ignore

# import wandb
# from src.data.make_dataset import load_data
from src.models.model import GCN

sys.path.append("..")



log = logging.getLogger(__name__)
print = log.info
# wandb.init(project="Project-MLOps", entity="Project-MLOps")



device = "cuda" if torch.cuda.is_available() else "cpu"

def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

def evaluate(model: nn.Module, data: torch_geometric.data.Data) -> float:
    """
    Evaluates model on data and returns accuracy.
    :param model: Model to be evaluated
    :param data: Data to evaluate on
    :return: accuracy
    """

    # model.eval()
    # out = model(data.x, data.edge_index)
    # # Use the class with highest probability.
    # pred = out.argmax(dim=1)
    # # Check against ground-truth labels.
    # test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # # Derive ratio of correct predictions.
    # test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    # return test_acc
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['user', 'movie'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['user', 'movie'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


@hydra.main(config_path="../config", config_name="default_config.yaml")
def train(self):    
    """
    Trains the model with hyperparameters in config on train data,
    saves the model and evaluates it on test data.
    :param config: Config file used for Hydra
    :return:
    """

    print("Training")
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment.hyperparams
    # wandb.config = hparams
    torch.manual_seed(hparams["seed"])
    orig_cwd = hydra.utils.get_original_cwd()

    # Load data
 #   data = load_data(orig_cwd + "/data/", name="Cora")
    train_data = torch.load('./data/processed/train.pt')
    val_data = torch.load('./data/processed/val.pt')
    test_data = torch.load('./data/processed/test.pt')

    # Model
    model = GCN(
        hidden_channels=hparams["hidden_channels"],
        num_features=hparams["num_features"],
        num_classes=hparams["num_classes"],
        dropout=hparams["dropout"],
    )

    model = model.to(device)

    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=5e-4)
  #  criterion = torch.nn.CrossEntropyLoss()
    epochs = hparams["epochs"]
    train_loss = []


    # Train model
    for epoch in range(epochs):
        # Clear gradients
        optimizer.zero_grad()
        # Perform a single forward pass
        out = model(dtrain_data.x_dict, train_data.edge_index_dict,
                 train_data['user', 'movie'].edge_label_index)
        target = train_data['user', 'movie'].edge_label
        # Compute the loss solely based on the training nodes
        # loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss = weighted_mse_loss(pred, target, weight)
        # Derive gradients
        loss.backward()
        # Update parameters based on gradients
        optimizer.step()
        # Append results
        train_loss.append(loss.item())
        # Compute rmse
        train_rmse = test(train_data)
        val_rmse = test(val_data)
        test_rmse = test(test_data)

        # print
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
          f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
 #       wandb.log({"Training loss": loss})



    # Save model
    directory = orig_cwd + "/models/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + hparams["checkpoint_name"]
    checkpoint = {'num_features': hparams["num_features"],
                  'num_classes': hparams["num_classes"],
                  'hidden_channels': hparams["hidden_channels"],
                  'dropout': hparams["dropout"],
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filename)    





if __name__ == "__main__":
    train()    