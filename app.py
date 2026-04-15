import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set(style="whitegrid")

DATA_PATH = os.path.join("marks predictor", "Student_Marks.csv")


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def train_model(data: pd.DataFrame) -> LinearRegression:
    X = data[["number_courses", "time_study"]]
    y = data["Marks"]
    model = LinearRegression()
    model.fit(X, y)
    return model


def compute_metrics(data: pd.DataFrame, model: LinearRegression) -> dict:
    X = data[["number_courses", "time_study"]]
    y = data["Marks"]
    predictions = model.predict(X)
    return {
        "Mean Absolute Error": mean_absolute_error(y, predictions),
        "Mean Squared Error": mean_squared_error(y, predictions),
        "Root Mean Squared Error": np.sqrt(mean_squared_error(y, predictions)),
        "R² Score": r2_score(y, predictions),
    }


def plot_scatter(data: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(x="time_study", y="Marks", data=data, ax=axes[0], color="#2a9d8f")
    axes[0].set_title("Study Hours vs Marks")
    axes[0].set_xlabel("Study Hours")
    axes[0].set_ylabel("Marks")

    sns.scatterplot(x="number_courses", y="Marks", data=data, ax=axes[1], color="#e76f51")
    axes[1].set_title("Number of Courses vs Marks")
    axes[1].set_xlabel("Number of Courses")
    axes[1].set_ylabel("Marks")

    st.pyplot(fig)


def plot_actual_vs_predicted(data: pd.DataFrame, model: LinearRegression):
    X = data[["number_courses", "time_study"]]
    y = data["Marks"]
    predictions = model.predict(X)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y, y=predictions, color="#264653", ax=ax)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "--", color="gray")
    ax.set_title("Actual vs Predicted Marks")
    ax.set_xlabel("Actual Marks")
    ax.set_ylabel("Predicted Marks")
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="Student Marks Predictor", page_icon=":mortar_board:", layout="wide")
    st.title("Student Marks Predictor")
    st.write(
        "Use this simple web app to estimate student marks based on the number of courses and weekly study hours. "
        "The model is trained with linear regression on the provided dataset."
    )

    data = load_data(DATA_PATH)
    model = train_model(data)
    metrics = compute_metrics(data, model)

    st.header("Dataset Overview")
    st.write(f"Loaded dataset with **{data.shape[0]}** rows and **{data.shape[1]}** columns.")
    st.dataframe(data.head())

    st.header("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        for name, value in metrics.items():
            st.metric(label=name, value=f"{value:.2f}")

    with col2:
        st.markdown(
            f"**Regression equation:**  
             Marks = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Courses + {model.coef_[1]:.2f} × Study Hours"
        )

    st.header("Visual Analysis")
    plot_scatter(data)
    plot_actual_vs_predicted(data, model)

    st.header("Predict Marks")
    st.write("Enter student inputs below and click `Predict` to estimate marks.")

    with st.form(key="prediction_form"):
        cols = st.columns(2)
        number_courses = cols[0].number_input("Number of Courses", min_value=0, max_value=12, value=3, step=1)
        study_hours = cols[1].slider("Study Hours per Week", min_value=0.0, max_value=16.0, value=5.0, step=0.5)
        submit_button = st.form_submit_button("Predict")

    if submit_button:
        predicted_marks = model.predict([[number_courses, study_hours]])[0]
        st.success(f"Predicted Marks: {predicted_marks:.1f}")
        st.info(
            f"Input summary: **{number_courses} courses** and **{study_hours:.1f} study hours per week**."
        )

    st.markdown("---")
    st.markdown(
        "**Note:** This app uses a simple linear regression model that works best within the observed data range. "
        "For stronger predictions, consider adding more features or using a more complex model."
    )


if __name__ == "__main__":
    main()
