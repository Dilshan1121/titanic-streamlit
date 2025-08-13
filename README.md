#  Titanic Survival Prediction – Streamlit App

An interactive **machine learning web application** built with [Streamlit](https://streamlit.io), designed to predict whether a Titanic passenger would survive — powered by a trained Random Forest model with visually engaging features.

---

##  Features

- **Data Exploration** – Inspect dataset overview, missing values, and preview rows.
- **Visualizations** – Interactive charts revealing survival trends by sex, age, and class.
- **Prediction Tool** – Input passenger attributes to receive a survival prediction and probability.
- **Model Performance** – View evaluation metrics and a confusion matrix.

---

##  Dataset

This app uses the classic Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic), which includes variables such as:

- `Survived` – target variable (0 = did not survive, 1 = survived)  
- `Pclass`, `Sex`, `Age`, `Fare`, `SibSp`, `Parch`, `Embarked`, `Title` (extracted from names)

---

##  Tech Stack

| Library          | Purpose                     |
|------------------|-----------------------------|
| Python 3         | Core programming language   |
| Pandas           | Data handling               |
| Scikit-learn     | Model training and evaluation |
| Plotly Express   | Dynamic and interactive charts |
| Streamlit        | App interface and deployment |
| Joblib           | Model serialization         |

---

##  Live App & Repo

- **Live Demo**: [Click here to try the app!](https://dilshan1121-titanic-streamlit-app-9zoiui.streamlit.app/)  
- **GitHub Repo**: https://github.com/Dilshan1121/titanic-streamlit

---

##  Installation (Run Locally)

1. Clone the repo:
    ```bash
    git clone https://github.com/Dilshan1121/titanic-streamlit.git
    cd titanic-streamlit
    ```

2. Set up virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate      # macOS / Linux
    venv\Scripts\activate         # Windows
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

Then open http://localhost:8501 in your browser.

---



