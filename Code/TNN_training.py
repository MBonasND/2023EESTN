import json
import random
from typing import Optional
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model import TimeSeriesForcasting
from training import split_df, Dataset


data_csv_path="DESNInputTNN.csv"
feature_target_names_path="configL96.json"
output_json_path="trained_config.json"
log_dir="ts_views_logs"
model_dir="ts_views_models"
horizon_size = 20


def pad_arr(arr: np.ndarray, expected_size: int = 981):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def df_to_np(df):
    arr = np.array(df)
    arr = pad_arr(arr)
    return arr


with open('configL96.json') as f:
    feature_target_names = json.load(f)

data_train = data[~data[feature_target_names["target"]].isna()]

grp_by_train = data_train.groupby(by=feature_target_names["group_by_key"])

groups = list(grp_by_train.groups)

full_groups = [
    grp for grp in groups if grp_by_train.get_group(grp).shape[0] > 2 * horizon
]


class PrintCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        print(f"Epoch: {trainer.current_epoch}, "
              f"Train Loss: {metrics['train_loss']:.4f}")

def train(
    data_csv_path: str,
    feature_target_names_path: str,
    output_json_path: str,
    log_dir: str = "ts_logs",
    model_dir: str = "ts_models",
    batch_size: int = 4,
    epochs: int = 50000,
    horizon_size: int = 20,
):
    data = pd.read_csv(data_csv_path)

    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)

    data_train = data[~data[feature_target_names["target"]].isna()]

    grp_by_train = data_train.groupby(by=feature_target_names["group_by_key"])

    groups = list(grp_by_train.groups)

    full_groups = [
        grp for grp in groups if grp_by_train.get_group(grp).shape[0] > 2 * horizon
    ]

    train_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
        split="train",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
    )
    val_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
        split="val",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
    )

    # print("len(train_data)", len(train_data))
    # print("len(val_data)", len(val_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
    )

    model = TimeSeriesForcasting(
        n_encoder_inputs=len(feature_target_names["features"]) + 1,
        n_decoder_inputs=len(feature_target_names["features"]) + 1,
        lr=1e-5,
        dropout=0.1,
    )

    logger = TensorBoardLogger(
        save_dir=log_dir,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=model_dir,
        filename="ts",
    )



    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator = "auto",
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
        enable_progress_bar = False
    )


    trainer.fit(model, train_loader, val_loader)

    print(trainer.logger.experiment)

    result_val = trainer.test(dataloaders=val_loader)

    output_json = {
        "val_loss": result_val[0]["test_loss"],
        "best_model_path": checkpoint_callback.best_model_path,
    }

    if output_json_path is not None:
        with open(output_json_path, "w") as f:
            json.dump(output_json, f, indent=4)

    return output_json

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("esn_for_tnn.csv")
# parser.add_argument("configL96.json")
# parser.add_argument("trained_config.json", default=None)
# parser.add_argument("ts_views_logs")
# parser.add_argument("ts_views_models")
# parser.add_argument("--epochs", type=int, default=2000)
# args = parser.parse_args()

# print(args)

if __name__ == "__main__":
    #import multiprocessing
    #multiprocessing.freeze_support()
    train(
        data_csv_path="DESNInputTNN.csv",
        feature_target_names_path="configL96.json",
        output_json_path="trained_config.json",
        log_dir="ts_views_logs",
        model_dir="ts_views_models",
        epochs = 50000,
        horizon_size = horizon_size
    )
    
    #print(multiprocessing. cpu_count())



    


