from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from modele import evaluate_model, load_data, predict, train_model

DATA_PATH = Path(__file__).parent / "donnees_lesions_synthetiques.csv"

st.set_page_config(page_title="POC lésions", layout="wide")
st.title("POC lésions")


@st.cache_data
def get_data():
    return load_data(DATA_PATH)


@st.cache_resource
def get_model(data):
    return train_model(data)


try:
    data = get_data()
    model, evaluation, _ = get_model(data)
except (OSError, ValueError) as error:
    st.error(f"Impossible de charger ou valider les données : {error}")
    st.stop()

metrics = evaluate_model(evaluation)

st.header("Estimation")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    diameter = col1.number_input("Diamètre (mm)", min_value=0.0, max_value=20.0, value=6.0, step=0.1)
    asymmetry = col2.slider("Asymétrie", min_value=0, max_value=5, value=2)
    color_variation = col3.slider("Variation de couleur", min_value=0, max_value=5, value=2)
    submitted = st.form_submit_button("Estimer")

if submitted:
    category, probability = predict(model, diameter, asymmetry, color_variation)
    st.subheader(f"Niveau de suspicion estimée : {category}")
    st.write(f"Confiance du modèle : **{probability:.1%}**")

st.header("Évaluation du modèle")
col1, col2, col3 = st.columns(3)
col1.metric("Observations", len(data))
col2.metric("Exactitude (test)", f"{metrics['accuracy']:.1%}")
col3.metric("F1 - catégorie Élevée", f"{metrics['f1']:.1%}")

st.subheader("Matrice de confusion")
figure, axis = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    evaluation["reel"],
    evaluation["prediction"],
    labels=["Faible", "Elevee"],
    display_labels=["Faible", "Elevee"],
    cmap="Blues",
    ax=axis,
)
axis.set_title("Prédictions du modèle")
st.pyplot(figure)
plt.close(figure)
