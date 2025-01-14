# German credit scoring

This project focuses on developing a machine learning model to evaluate the creditworthiness of clients using the [German Credit Data dataset](https://www.kaggle.com/datasets/elsnkazm/german-credit-scoring-data). The goal is to predict whether a borrower is a good or bad credit risk, assisting financial institutions in making loan approval decisions.

## Live Application
The project includes a live interactive web application built with Streamlit, which allows users to explore the model and test its predictions. You can access the application here:
https://german-credit-scoring.streamlit.app

## Dataset
This project uses the [German Credit Data dataset](https://www.kaggle.com/datasets/elsnkazm/german-credit-scoring-data), which contains information about 1,000 borrowers and 20 variables describing their characteristics.

## Tools and Technologies
During the project, several tools and technologies were used, including:

 - **MLflow**: For tracking experiments, managing models, and facilitating reproducibility throughout the machine learning workflow.
 - **Python**: For data processing, modeling, and application development.
 - **Streamlit**: To create an interactive web application for showcasing the model.
 - **Docker**: To containerize the application for consistent deployment across environments.
 - **Cookiecutter Data Science**: For structuring the project following best practices for data science workflows.

## Project Structure
`data/raw/`: Original raw data.

`models/`: Saved model and params.

`notebooks/`: Jupyter notebooks used for exploratory data analysis (EDA), training a baseline model, hyperparameter tuning and training the final XGBoost model.

`streamlit/`: Files for the Streamlit web application.

`config.json`: Configuration parameters for the project.

`requirements.txt`: Python dependencies.

`Dockerfile`: Instructions to build the Docker image.

## Requirements
 - Python 3.12
 - Dependencies listed in `requirements.txt`


