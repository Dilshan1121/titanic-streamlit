import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px
import os
import time
from PIL import Image  

st.set_page_config(
    page_title="🚢 Titanic Survival Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

def add_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f7f9fc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image()

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/Titanic-Dataset.csv")

model = load_model()
df = load_data()

def animated_metric(label, value, duration=1):
    placeholder = st.empty()
    for i in range(int(value) + 1):
        placeholder.metric(label, i)
        time.sleep(duration / value if value != 0 else 0)

def load_local_image(image_filename):
    image_path = os.path.join("images", image_filename)
    if os.path.exists(image_path):
        return Image.open(image_path)
    return None

sidebar_img = load_local_image("titanic_sidebar.jpg")
if sidebar_img:
    st.sidebar.image(sidebar_img, use_container_width=True)

st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Page", ["Home", "Data Exploration", "Visualizations", "Prediction", "Model Performance"])

if page == "Home":
    st.title("🚢 Titanic Survival Prediction App")

    st.markdown("""
        Welcome to the Titanic Survival Prediction App!  
        Use the navigation menu on the left to explore the dataset, view visualizations, make predictions, and check model performance.
    """)
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

elif page == "Data Exploration":
    st.header("📊 Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        animated_metric("Rows", df.shape[0])
    with col2:
        animated_metric("Columns", df.shape[1])
    with col3:
        animated_metric("Missing Values", int(df.isnull().sum().sum()))

    st.write("### Missing Values by Column")
    st.bar_chart(df.isnull().sum())

    st.subheader("Random Sample from Data")
    sample_df = df.sample(10)
    st.dataframe(sample_df)

    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download This Sample as CSV",
        data=csv,
        file_name='titanic_sample.csv',
        mime='text/csv'
    )

elif page == "Visualizations":
    st.header("📈 Visual Insights")
    col1, col2 = st.columns(2)

    surv_by_sex = df.groupby('Sex')['Survived'].mean().reset_index()
    fig1 = px.bar(surv_by_sex, x='Sex', y='Survived', title="Survival Rate by Sex", color='Sex')
    col1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x='Age', nbins=25, title="Age Distribution", marginal="box", color='Survived')
    col2.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(df, x='Pclass', y='Fare', points="outliers", title="Fare by Passenger Class", color='Pclass')
    st.plotly_chart(fig3, use_container_width=True)

elif page == "Prediction":
    st.header("🎯 Make a Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Pclass", [1, 2, 3])
            sex = st.selectbox("Sex", ["male", "female"])
            age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
            sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
        with col2:
            parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
            fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
            embarked = st.selectbox("Embarked", ["C", "Q", "S"])
            title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Other"])

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
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

        if pred == 1:
            st.success(f"✅ Passenger Survived (Probability: {proba:.2f})")
        else:
            st.error(f"❌ Passenger Did NOT Survive (Survival Probability: {proba:.2f})")

elif page == "Model Performance":
    st.header("📊 Model Performance")
    if os.path.exists("artifacts/metrics.json"):
        with open("artifacts/metrics.json") as f:
            metrics = json.load(f)
        st.json(metrics)
    if os.path.exists("artifacts/confusion_matrix.png"):
        st.image("artifacts/confusion_matrix.png", caption="Confusion Matrix")
