import pandas as pd
import xgboost as xgb
import mlflow
import json
import pickle
import copy

from sklearn.model_selection import train_test_split
from pathlib import Path


def to_category(data):
    data = copy.deepcopy(data)
    for c in data.columns:
        col_type = data[c].dtype
        if (
            col_type == "object"
            or col_type.name == "category"
            or col_type.name == "datetime64[ns]"
            or col_type.name == "string"
            or col_type == "string"
        ):
            data[c] = data[c].astype("category")

    return data


def convert(value):
    try:
        return int(value)
    except ValueError:
        return float(value)


config_path = Path("config.json").resolve()
with open(config_path, "r") as file:
    config = json.load(file)

RUN_ID = config["run_id"]
EXP_ID = config["experiment_id"]
MODEL_URI = f"mlflow-artifacts:/{EXP_ID}/{RUN_ID}/artifacts/XGBClassifier"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("german-credit-scoring")
run_data = mlflow.get_run(RUN_ID).data.to_dictionary()

with open("models/params.json", "w") as outfile:
    json.dump(run_data["params"], outfile)

data_path = Path("data/raw/german_credit_cleaned.csv").resolve()
data = pd.read_csv(data_path)
data["target"] = data["target"] == "good"
data["target"] = data["target"].astype("int")
X = data.drop(columns=["target"])
y = data["target"]

model = xgb.XGBClassifier(enable_categorical=True, random_state=42)
model.fit(to_category(X), y)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

