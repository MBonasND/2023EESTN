import json
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
from training import split_df


data_csv_path = "DESNInputTNNTest.csv"
feature_target_names_path="configL96.json"
eval_json_path = 'eval.json'
data_for_visualization_path = 'visualization.json'
trained_json_path = 'trained_config.json'
horizon = 20
history_shown = 10




def pad_arr(arr: np.ndarray, expected_size: int = 941):
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



class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, split, features, target):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.features = features
        self.target = target

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)

        src, trg = split_df(df, split=self.split, history_size=960, horizon_size=20)

        src = src[self.features + [self.target]]

        src = df_to_np(src)

        trg_in = trg[self.features + [f"{self.target}_lag_0"]]

        trg_in = np.array(trg_in)
        trg_out = np.array(trg[self.target])

        src = torch.tensor(src, dtype=torch.float)
        trg_in = torch.tensor(trg_in, dtype=torch.float)
        trg_out = torch.tensor(trg_out, dtype=torch.float)

        return src, trg_in, trg_out


def smape(true, pred):
 
    true = np.array(true)
    pred = np.array(pred)


    smape_val = (
        np.mean((true - pred)**2)
    )

    return smape_val


def evaluate_regression(true, pred):
    """
    eval mae + smape
    :param true:
    :param pred:
    :return:
    """

    return {"smape": smape(true, pred), "mae": mean_absolute_error(true, pred)}


def evaluate(
    data_csv_path: str,
    feature_target_names_path: str,
    trained_json_path: str,
    eval_json_path: str,
    horizon_size: int = 20,
    data_for_visualization_path: Optional[str] = None,
):
    """
    Evaluates the model on the last 8 labeled weeks of the data.
    Compares the model to a simple baseline : prediction the last known value
    :param data_csv_path:
    :param feature_target_names_path:
    :param trained_json_path:
    :param eval_json_path:
    :param horizon_size:
    :param data_for_visualization_path:
    :return:
    """
    data = pd.read_csv(data_csv_path)

    with open(trained_json_path) as f:
        model_json = json.load(f)

    model_path = model_json["best_model_path"]

    with open(feature_target_names_path) as f:
        feature_target_names = json.load(f)

    target = feature_target_names["target"]

    data_train = data[~data[target].isna()]

    grp_by_train = data_train.groupby(by=feature_target_names["group_by_key"])

    groups = list(grp_by_train.groups)

    full_groups = [
        grp for grp in groups if grp_by_train.get_group(grp).shape[0] > horizon
    ]

    val_data = Dataset(
        groups=full_groups,
        grp_by=grp_by_train,
        split="test",
        features=feature_target_names["features"],
        target=feature_target_names["target"],
    )

    src, trg_in, _ = val_data[0]
    src, trg_in = src.unsqueeze(0), trg_in.unsqueeze(0)

    model = TimeSeriesForcasting(
        n_encoder_inputs=len(feature_target_names["features"]) + 1,
        n_decoder_inputs=len(feature_target_names["features"]) + 1,
        lr=1e-5,
        dropout=0.5,
    )
    model.load_state_dict(torch.load(model_path)["state_dict"])

    model.eval()

    #prediction = model((src, trg_in))

    gt = []
    baseline_last_known_values = []
    neural_predictions = []

    data_for_visualization = []


    for i, group in tqdm(enumerate(full_groups)):
        time_series_data = {"history": [], "ground_truth": [], "prediction": []}

        df = grp_by_train.get_group(group)
        src, trg = split_df(df, split="test", history_size=960, horizon_size=20)

        time_series_data["history"] = src[target].tolist()[-history_shown:]
        time_series_data["ground_truth"] = trg[target].tolist()

        last_known_value = src[target].values[-1]

        trg["last_known_value"] = last_known_value


        gt += trg[target].tolist()
        baseline_last_known_values += trg["last_known_value"].tolist()

        src, trg_in, _ = val_data[i]

        src, trg_in = src.unsqueeze(0), trg_in.unsqueeze(0)

        with torch.no_grad():
            prediction = model((src, trg_in))
            
            trg[target + "_predicted"] = (prediction.squeeze().numpy()).tolist()

            neural_predictions += trg[target + "_predicted"].tolist()

            time_series_data["prediction"] = trg[target + "_predicted"].tolist()

        data_for_visualization.append(time_series_data)



    # print(len(neural_predictions))
    # print(len(gt))
    # print(baseline_last_known_values)

    baseline_eval = evaluate_regression(gt, baseline_last_known_values)
    model_eval = evaluate_regression(gt, neural_predictions)

    eval_dict = {
        "Baseline_MAE": baseline_eval["mae"],
        "Baseline_SMAPE": baseline_eval["smape"],
        "Model_MAE": model_eval["mae"],
        "Model_SMAPE": model_eval["smape"],
    }

# print(eval_dict)

    if eval_json_path is not None:
        with open(eval_json_path, "w") as f:
            json.dump(eval_dict, f, indent=4)

    if data_for_visualization_path is not None:
        with open(data_for_visualization_path, "w") as f:
            json.dump(data_for_visualization, f, indent=4)

    for k, v in eval_dict.items():
        print(k, v)

    return eval_dict


if __name__ == "__main__":
    #import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_csv_path")
    # parser.add_argument("--feature_target_names_path")
    # parser.add_argument("--trained_json_path")
    # parser.add_argument("--eval_json_path", default=None)
    # parser.add_argument("--data_for_visualization_path", default=None)
    # args = parser.parse_args()

    evaluate(
        data_csv_path=data_csv_path,
        feature_target_names_path=feature_target_names_path,
        trained_json_path=trained_json_path,
        eval_json_path=eval_json_path,
        data_for_visualization_path=data_for_visualization_path,
    )