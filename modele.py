from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(csv_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def train_model(data: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    x_train, x_test, y_train, y_test = train_test_split(
        data[["diametre_mm", "asymetrie", "variation_couleur"]],
        data["categorie_fictive"],
        test_size=0.2,
        random_state=42,
        stratify=data["categorie_fictive"],
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    evaluation = pd.DataFrame({"reel": y_test, "prediction": predictions})
    return model, evaluation, x_test


def evaluate_model(evaluation: pd.DataFrame) -> dict:
    y_test = evaluation["reel"]
    predictions = evaluation["prediction"]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, pos_label="Elevee"),
        "matrix": confusion_matrix(y_test, predictions, labels=["Faible", "Elevee"]),
    }


def predict(model: Pipeline, diameter: float, asymmetry: int, color_variation: int) -> tuple[str, float]:
    sample = pd.DataFrame(
        [[diameter, asymmetry, color_variation]],
        columns=["diametre_mm", "asymetrie", "variation_couleur"],
    )
    category = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    class_index = list(model.classes_).index(category)
    return category, float(probabilities[class_index])
