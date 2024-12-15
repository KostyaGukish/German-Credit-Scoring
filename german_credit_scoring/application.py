import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import mlflow
import copy
import shap

from pathlib import Path
from ydata_profiling import ProfileReport


import eda

RUN_ID = "ec47c80d2a21439dbc0e7a8dbdecfb54"
MODEL_URI = f"mlflow-artifacts:/692688493119941248/{RUN_ID}/artifacts/XGBClassifier"


def convert(value):
    try:
        return int(value)
    except ValueError:
        return float(value)


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("german-credit-scoring")
model = mlflow.xgboost.load_model(MODEL_URI)
run_data = mlflow.get_run(RUN_ID).data.to_dictionary()
threshold = float(run_data["params"]["threshold"])
run_data["params"].pop("threshold")
params = run_data["params"]
params = {key: convert(value) for key, value in params.items()}
shap_model = xgb.XGBClassifier(enable_categorical=True, random_state=42, **params)


@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")


@st.cache_data
def show_profile(data):
    data_path = Path("data/raw/german_credit_cleaned.csv").resolve()
    data = pd.read_csv(data_path)
    profile = ProfileReport(data, title="Profiling Report")
    html_content = profile.to_html()
    st.components.v1.html(html_content, height=800, scrolling=True)


@st.cache_data
def show_numerical_statistics(data):
    for c in eda.NUM_COLS:
        fig1, fig2 = eda.numerical_plot(data, c, 20)
        st.subheader(c)
        st.plotly_chart(fig1, theme=None, use_container_width=True)
        st.plotly_chart(fig2, theme=None, use_container_width=True)

    for c in eda.NUM_COLS:
        st.subheader(c)
        fig = eda.box_plot(data, c)
        st.plotly_chart(fig, theme=None, use_container_width=True)


@st.cache_data
def show_categorical_statistics(data):
    for c in eda.CAT_COLS:
        fig = eda.get_percentages(data, c)
        st.subheader(c)
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
def explane_data(uploaded_data):
    explainer = shap.TreeExplainer(shap_model)
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
def explane_row(uploaded_data, row):
    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer(to_category(uploaded_data))

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[row], max_display=25)
    st.pyplot(fig)


st.set_page_config(layout="wide", page_title="German credit scoring")
st.title("German credit scoring")


data_path = Path("data/raw/german_credit_cleaned.csv").resolve()
data = pd.read_csv(data_path)
data["target"] = data["target"] == "good"
data["target"] = data["target"].astype("int")
X = data.drop(columns=["target"])
y = data["target"]
shap_model.fit(to_category(X), y)

st.sidebar.header("Upload and download :gear:")
uploaded_file = st.sidebar.file_uploader("Upload a data", type=["csv"])

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)
    uploaded_data = uploaded_data.drop(columns="target")

    predicted_data = copy.deepcopy(uploaded_data)
    predicted_data = predict(uploaded_data)
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
        explane_data(uploaded_data)
        
    with shap2:
        if (row != -1):
            explane_row(uploaded_data, row)
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
