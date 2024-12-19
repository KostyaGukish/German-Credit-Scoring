import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import shap
import json
import pickle

from pathlib import Path
from ydata_profiling import ProfileReport

import eda


config_path = Path("./config.json").resolve()
with open(config_path, "r") as file:
    config = json.load(file)


@st.cache_resource
def load_model():
    model_info_path = Path("./models").resolve()
    model_path = model_info_path / "model.pkl"
    params_path = model_info_path / "params.json"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(params_path, "r") as file:
        params = json.load(file)

    threshold = float(params["threshold"])
    return model, threshold


def convert(value):
    try:
        return int(value)
    except ValueError:
        return float(value)


@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")


@st.cache_data
def show_profile(data):
    profile = ProfileReport(data, title="Profiling Report")
    html_content = profile.to_html()
    st.components.v1.html(html_content, height=800, scrolling=True)


@st.cache_data
def show_numerical_statistics(data):
    for c in eda.NUM_COLS:
        fig1, fig2 = eda.numerical_plot(data, c, 20)
        st.subheader(config["features"][c]["info"])
        st.plotly_chart(fig1, theme=None, use_container_width=True)
        st.plotly_chart(fig2, theme=None, use_container_width=True)

        st.subheader(config["features"][c]["info"])
        fig = eda.box_plot(data, c)
        st.plotly_chart(fig, theme=None, use_container_width=True)


@st.cache_data
def show_categorical_statistics(data):
    for c in eda.CAT_COLS:
        fig = eda.get_percentages(data, c)
        st.subheader(config["features"][c]["info"])
        st.plotly_chart(fig, theme=None, use_container_width=True)


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


def check(data):
    pass


@st.cache_data
def predict(data):
    predictions = model.predict_proba(to_category(data))[:, 1]
    predictions = np.where(predictions > threshold, "good", "bad")

    data["target"] = predictions
    return data


@st.cache_data
def explain_data(uploaded_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(to_category(uploaded_data))
    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    shap.plots.heatmap(shap_values, instance_order=shap_values.sum(1))
    st.pyplot(fig)


@st.cache_data
def explain_row(uploaded_data, row):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(to_category(uploaded_data))

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[row], max_display=25)
    st.pyplot(fig)


st.set_page_config(layout="wide", page_title="German credit scoring")
st.title("German credit scoring")
st.sidebar.header("Upload and download :gear:")

model, threshold = load_model()

uploaded_file = st.sidebar.file_uploader("Upload a data", type=["csv"])

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file).drop(columns="target")

    predicted_data = copy.deepcopy(uploaded_data)
    predicted_data = predict(predicted_data)
    st.header("My data :books:")

    event = st.dataframe(
        predicted_data,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    if len(event.selection.rows) == 1:
        row = event.selection.rows[0]
    else:
        row = -1

    st.header("Explanation")
    shap1, shap2 = st.tabs(["Entire data", "One row"])

    with shap1:
        explain_data(uploaded_data)

    with shap2:
        if row != -1:
            explain_row(uploaded_data, row)
        else:
            st.markdown("No rows selected.")

    st.sidebar.markdown("\n")
    st.sidebar.download_button(
        "Download processed data :floppy_disk:",
        convert_df(predicted_data),
        "processed.csv",
        mime="text/csv",
    )

st.header("Data Statistics :bar_chart:")

data_path = Path("data/raw/german_credit_cleaned.csv").resolve()
data = pd.read_csv(data_path)
data["target"] = data["target"] == "good"
data["target"] = data["target"].astype("int")

tab1, tab2, tab3 = st.tabs(
    [
        "Profiling report",
        "Numerical columns statistics",
        "Categorical columns statistics",
    ]
)
with tab1:
    show_profile(data)
with tab2:
    show_numerical_statistics(data)
with tab3:
    show_categorical_statistics(data)
