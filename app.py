import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import os

st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/Titanic-Dataset.csv")

model = load_model()
df = load_data()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualizations", "Prediction", "Model Performance"])

if page == "Home":
    st.title("🚢 Titanic Survival Prediction App")
    st.write("""
    This app predicts whether a passenger survived the Titanic disaster based on selected attributes.
    """)
    st.dataframe(df.head())

elif page == "Data Exploration":
    st.header("Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Missing values:", df.isnull().sum())
    st.write(df.dtypes)
    st.subheader("Sample Data")
    st.dataframe(df.sample(10))

elif page == "Visualizations":
    st.header("Visualizations")
    surv_by_sex = df.groupby('Sex')['Survived'].mean().reset_index()
    fig1 = px.bar(surv_by_sex, x='Sex', y='Survived', title="Survival Rate by Sex")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x='Age', nbins=25, title="Age Distribution", marginal="box")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(df, x='Pclass', y='Fare', points="outliers", title="Fare by Passenger Class")
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Prediction":
    st.header("Make a Prediction")
    pclass = st.selectbox("Pclass", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
    embarked = st.selectbox("Embarked", ["C", "Q", "S"])
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"])

    if st.button("Predict"):
        input_df = pd.DataFrame([{
            "Pclass": pclass,
            "Sex": sex,
            "Age": age,
            "SibSp": sibsp,
            "Parch": parch,
            "Fare": fare,
            "Embarked": embarked,
            "Title": title
        }])
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        st.success(f"Prediction: {'Survived' if pred==1 else 'Did not survive'}")
        st.info(f"Survival Probability: {proba:.2f}")

elif page == "Model Performance":
    st.header("Model Performance")
    if os.path.exists("artifacts/metrics.json"):
        with open("artifacts/metrics.json") as f:
            metrics = json.load(f)
        st.write(metrics)
    if os.path.exists("artifacts/confusion_matrix.png"):
        st.image("artifacts/confusion_matrix.png")
