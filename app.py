import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(page_title="AIDS Clinical Trial Classification", layout="wide")

st.title("AIDS Clinical Trial Classification App")

st.markdown("Upload test CSV data and select a trained model to predict.")

# Load Models
models = {
    "Logistic Regression": joblib.load("model/Logistic Regression.pkl"),
    "Decision Tree": joblib.load("model/Decision Tree.pkl"),
    "KNN": joblib.load("model/KNN.pkl"),
    "Naive Bayes": joblib.load("model/Naive Bayes.pkl"),
    "Random Forest": joblib.load("model/Random Forest.pkl"),
    "XGBoost": joblib.load("model/XGBoost.pkl")
}

scaler = joblib.load("model/scaler.pkl")

# Upload file
uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_name = st.selectbox("Select Model", list(models.keys()))

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(data.head())

    # IMPORTANT: change target column name
    target_column = "infected"

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_scaled = scaler.transform(X)

    model = models[model_name]

    predictions = model.predict(X_scaled)

    st.subheader("Evaluation Metrics")

    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(accuracy_score(y, predictions), 3))
    col2.metric("Precision", round(precision_score(y, predictions), 3))
    col3.metric("Recall", round(recall_score(y, predictions), 3))

    col1.metric("F1 Score", round(f1_score(y, predictions), 3))
    col2.metric("MCC", round(matthews_corrcoef(y, predictions), 3))
    col3.metric("AUC", round(roc_auc_score(y, predictions), 3))

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(classification_report(y, predictions))
