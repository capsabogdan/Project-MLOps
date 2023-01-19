import os
import hydra
import torch
import time
import torch.nn.functional as F
from google.cloud import storage 
from omegaconf import DictConfig, OmegaConf
from src.models.model import Model


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)

    return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()


def push_data_to_cloud(bucket_name, file):
    print("Downloading data...")
    storage_client = storage.Client(project="zeroshots")
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(file)
    blob.download_to_filename(file)


def fetch_data_from_cloud(bucket_name, file_names):
    path_to_download = "src/models/temp_data/"
    print("Downloading data...")
    storage_client = storage.Client(project="zeroshots")
    bucket = storage_client.bucket(bucket_name)

    if not os.path.exists(path_to_download):
        os.mkdir(path_to_download)

    for file in file_names:
        blob = bucket.blob(file)
        blob.download_to_filename(path_to_download + file)

    print("Download complete! Sleeping for 5")
    
    time.sleep(5)


def read_data():
    # Load data
    print("thomas", os.getcwd())
    train_data = torch.load("src/models/temp_data/train.pt")
    val_data = torch.load("src/models/temp_data/val.pt")
    test_data = torch.load("src/models/temp_data/test.pt")

    return train_data, test_data, val_data


def push_model_to_cloud(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client(project='zeroshots')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )    



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

    device = "cpu"  #"cuda" if torch.cuda.is_available() else "cpu"
    train_data, test_data, val_data = read_data()

    print("Training")
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiments.hyperparams
    # wandb.config = hparams
    torch.manual_seed(hparams["seed"])

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
#       wandb.log({"Training loss": loss})

    orig_cwd = hydra.utils.get_original_cwd()

    # Save model
    directory = orig_cwd + "/models/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + hparams["checkpoint_name"]
    checkpoint = {'hidden_channels': hparams["hidden_channels"],
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, filename)  

    push_model_to_cloud('movie-rec-model-checkpoints', filename, hparams["checkpoint_name"])


if __name__ == "__main__":
    fetch_data_from_cloud("movies-mlops-clean", ["train.pt", "test.pt", "val.pt"])
    train()
