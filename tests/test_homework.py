# flake8: noqa: E501
"""Autograding script."""

import json
import os
import pickle

import pandas as pd  # type: ignore

# ------------------------------------------------------------------------------
MODEL_FILENAME = "files/models/model.pkl"
MODEL_COMPONENTS = [
    "OneHotEncoder",
    "SelectKBest",
    "MinMaxScaler",
    "LogisticRegression",
]
SCORES = [
    0.639,
    0.654,
]
METRICS = [
    {
        "type": "metrics",
        "dataset": "train",
        "precision": 0.693,
        "balanced_accuracy": 0.639,
        "recall": 0.319,
        "f1_score": 0.437,
    },
    {
        "type": "metrics",
        "dataset": "test",
        "precision": 0.701,
        "balanced_accuracy": 0.654,
        "recall": 0.349,
        "f1_score": 0.466,
    },
    {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": 15560, "predicted_1": None},
        "true_1": {"predicted_0": None, "predicted_1": 1508},
    },
    {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": 6785, "predicted_1": None},
        "true_1": {"predicted_0": None, "predicted_1": 660},
    },
]


def split_df(df):
    """User function"""
    # Prepare the data
    df = df.loc[(df["EDUCATION"] != 0)]
    df = df.loc[(df["MARRIAGE"] != 0)]
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4

    # Split the data
    selected_columns = [
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    x_df = df[selected_columns]
    y_df = df["default payment next month"]
    return x_df, y_df


# ------------------------------------------------------------------------------
#
# Internal tests
#


# ------------------------------------------------------------------------------
#
# Internal tests
#
def _load_model():
    """Generic test to load a model"""
    assert os.path.exists(MODEL_FILENAME)
    with open(MODEL_FILENAME, "rb") as file:
        model = pickle.load(file)
    assert model is not None
    return model


def _test_components(model):
    """Test components"""
    assert "GridSearchCV" in str(type(model))
    current_components = [str(model.estimator[i]) for i in range(len(model.estimator))]
    for component in MODEL_COMPONENTS:
        assert any(component in x for x in current_components)


def _test_scores(model, x_train, y_train, x_test, y_test):
    """Test scores"""
    assert model.score(x_train, y_train) > SCORES[0]
    assert model.score(x_test, y_test) > SCORES[1]


def _load_metrics():
    assert os.path.exists("files/output/metrics.json")
    metrics = []
    with open("files/output/metrics.json", "r", encoding="utf-8") as file:
        for line in file:
            metrics.append(json.loads(line))
    return metrics


def _test_metrics(metrics):

    for index in [0, 1]:
        assert metrics[index]["type"] == METRICS[index]["type"]
        assert metrics[index]["dataset"] == METRICS[index]["dataset"]
        assert metrics[index]["precision"] > METRICS[index]["precision"]
        assert metrics[index]["balanced_accuracy"] > METRICS[index]["balanced_accuracy"]
        assert metrics[index]["recall"] > METRICS[index]["recall"]
        assert metrics[index]["f1_score"] > METRICS[index]["f1_score"]

    for index in [2, 3]:
        assert metrics[index]["type"] == METRICS[index]["type"]
        assert metrics[index]["dataset"] == METRICS[index]["dataset"]
        assert (
            metrics[index]["true_0"]["predicted_0"]
            > METRICS[index]["true_0"]["predicted_0"]
        )
        assert (
            metrics[index]["true_1"]["predicted_1"]
            > METRICS[index]["true_1"]["predicted_1"]
        )


def test_homework():
    """Tests"""

    model = _load_model()
    _test_components(model)

    train_df = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

    x_train, y_train = split_df(train_df)
    x_test, y_test = split_df(test_df)

    _test_scores(model, x_train, y_train, x_test, y_test)

    metrics = _load_metrics()
    _test_metrics(metrics)
