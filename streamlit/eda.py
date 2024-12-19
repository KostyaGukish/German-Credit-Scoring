import numpy as np
import math
import pandas as pd
from copy import deepcopy

import plotly.graph_objects as go
import plotly.express as px

import json

with open("./config.json", "r") as file:
    config = json.load(file)

CAT_COLS = [
    "checking_acc_status",
    "cred_hist",
    "purpose",
    "saving_acc_bonds",
    "present_employment_since",
    "personal_stat_gender",
    "other_debtors_guarantors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "is_foreign_worker",
    "installment_rate",
    "present_residence_since",
    "num_curr_loans",
    "num_people_provide_maint",
]

NUM_COLS = [
    "duration",
    "loan_amt",
    # "installment_rate", # 4 unique
    # "present_residence_since", # 4 unique
    "age",
    # "num_curr_loans", # 4 unique
    # "num_people_provide_maint", # 2 unique
]

line_name = "Probability of 'good' outcomes for the entire dataset"
bar_name = "Probability of 'good' outcomes for the column"


def get_percentages(data, column):
    temp = deepcopy(data)

    temp[column] = temp[column].astype(str)

    good = data[data["target"] == 1].shape[0] / data.shape[0]

    temp = temp.groupby(by=[column, "target"]).target.agg(["count"]).reset_index()
    temp = pd.merge(
        temp, temp.groupby(by=[column])["count"].agg(["sum"]).reset_index(), on=[column]
    )
    temp = temp[temp["target"] == 1]
    temp["value"] = temp["count"] / temp["sum"]

    temp = temp.sort_values("value", ascending=False)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=temp[column],
            y=temp["value"],
            name=bar_name,
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=temp[column], y=[good] * len(temp[column]), mode="lines", name=line_name
        )
    )

    fig.update_layout(
        yaxis=dict(title=dict(text="Probability")),
        xaxis=dict(title=dict(text=config["features"][column]["info"])),
    )

    return fig


def numerical_plot(data, column, n_bins):
    def count_elements_in_intervals(a, size, len, mn, mx):
        interval_counts = {}

        for ind in range(0, len):
            interval_counts[ind] = 0

        for num in a:
            interval_index = (num - mn + 1) // size
            interval_counts[interval_index] += 1

        sorted_counts = dict(sorted(interval_counts.items()))
        return pd.Series(sorted_counts)

    good = data[data["target"] == 1].shape[0] / data.shape[0]
    data_good = data[data["target"] == 1][column]
    data_bad = data[data["target"] == 0][column]
    size = math.ceil((data[column].max() - data[column].min()) / n_bins)
    len = math.ceil(data[column].max() / size)

    good_hist = count_elements_in_intervals(
        data_good, size, len, data[column].min(), data[column].max()
    )
    bad_hist = count_elements_in_intervals(
        data_bad, size, len, data[column].min(), data[column].max()
    )

    result = good_hist / (good_hist + bad_hist)
    result = result[:n_bins]

    left = (np.array(result.index) * size + data[column].min()).astype(str)
    right = ((np.array(result.index) + 1) * size - 1 + data[column].min()).astype(str)
    x = np.char.add(left, "-")
    x = np.char.add(x, right)

    fig1 = go.Figure()

    fig1.add_trace(
        go.Histogram(
            name="good",
            x=data_good,
            xbins=dict(
                start=data[column].min(), end=data[column].max() + size, size=size
            ),
        )
    )

    fig1.add_trace(
        go.Histogram(
            name="bad",
            x=data_bad,
            xbins=dict(
                start=data[column].min(), end=data[column].max() + size, size=size
            ),
        )
    )

    fig1.update_layout(
        height=550,
        width=1150,
        yaxis=dict(title=dict(text="Frequency")),
        xaxis=dict(title=dict(text=config["features"][column]["info"])),
    )

    fig2 = go.Figure()

    fig2.add_trace(
        go.Bar(x=x, y=result.values, name=bar_name),
    )

    fig2.add_trace(go.Scatter(x=x, y=[good] * x.size, mode="lines", name=line_name))

    fig2.update_layout(
        width=1150,
        yaxis=dict(title=dict(text="Probability")),
        xaxis=dict(title=dict(text=config["features"][column]["info"])),
    )

    return fig1, fig2


def box_plot(data, column):
    fig = px.box(data, x="target", y=column)
    return fig
