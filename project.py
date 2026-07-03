"""
Data-Level vs. Algorithm-Level Solutions for Imbalanced Classification

Run:
    python project.py

Expected file:
    creditcard.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
except ModuleNotFoundError as error:
    raise SystemExit(
        "Missing package: imbalanced-learn\n"
        "Install it with: python -m pip install imbalanced-learn"
    ) from error

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
CREDIT_MAJORITY_SAMPLE = 25000

FIGURE_DIR = Path("figures")
RESULTS_DIR = Path("results")
FIGURE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

np.random.seed(RANDOM_STATE)

METHOD_COLORS = {
    "Baseline": "tab:gray",
    "Random Oversampling": "tab:blue",
    "Random Undersampling": "tab:orange",
    "SMOTE": "tab:green",
    "Class Weight": "tab:red",
}


# ----------------------------- Data loading -----------------------------

def load_credit_card_data(path="creditcard.csv"):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "creditcard.csv was not found. Put it in the same folder as project.py."
        )

    data = pd.read_csv(path)

    if "Class" not in data.columns:
        raise ValueError("The credit card dataset must contain a column named 'Class'.")

    X = data.drop(columns=["Class"])
    y = data["Class"].astype(int)
    return X, y


def load_breast_cancer_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)  # 0 = malignant, 1 = benign
    return X, y


def class_summary(y, name):
    counts = pd.Series(y).value_counts().sort_index()
    majority = counts.max()
    minority = counts.min()

    return {
        "Dataset": name,
        "Class 0": int(counts.get(0, 0)),
        "Class 1": int(counts.get(1, 0)),
        "Imbalance Ratio": round(majority / minority, 2),
    }


def sample_credit_training_set(X_train, y_train):
    train_data = X_train.copy()
    train_data["Class"] = y_train.values

    fraud = train_data[train_data["Class"] == 1]
    legitimate = train_data[train_data["Class"] == 0]

    n_legitimate = min(CREDIT_MAJORITY_SAMPLE, len(legitimate))
    legitimate_sample = legitimate.sample(n=n_legitimate, random_state=RANDOM_STATE)

    sampled = pd.concat([fraud, legitimate_sample], axis=0)
    sampled = sampled.sample(frac=1, random_state=RANDOM_STATE)

    X_sampled = sampled.drop(columns=["Class"])
    y_sampled = sampled["Class"].astype(int)
    return X_sampled, y_sampled


# ----------------------------- Modeling -----------------------------

def make_model(model_name, method_name):
    class_weight = "balanced" if method_name == "Class Weight" else None

    if model_name == "Logistic Regression":
        return LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )

    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=35,
            max_depth=12,
            min_samples_leaf=2,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown model: {model_name}")


def resample_training_data(X_train, y_train, method_name):
    if method_name == "Random Oversampling":
        sampler = RandomOverSampler(random_state=RANDOM_STATE)
        return sampler.fit_resample(X_train, y_train)

    if method_name == "Random Undersampling":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
        return sampler.fit_resample(X_train, y_train)

    if method_name == "SMOTE":
        minority_count = int(pd.Series(y_train).value_counts().min())
        k_neighbors = max(1, min(3, minority_count - 1))
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
        return sampler.fit_resample(X_train, y_train)

    return X_train, y_train


def prepare_data_for_model(X_train, X_test, y_train, model_name, method_name):
    if model_name == "Logistic Regression":
        scaler = StandardScaler()
        X_train_used = scaler.fit_transform(X_train)
        X_test_used = scaler.transform(X_test)
    else:
        X_train_used = X_train
        X_test_used = X_test

    X_train_used, y_train_used = resample_training_data(
        X_train_used,
        y_train,
        method_name,
    )

    return X_train_used, X_test_used, y_train_used


def evaluate_model(dataset_name, X_train, X_test, y_train, y_test, model_name, method_name):
    X_train_used, X_test_used, y_train_used = prepare_data_for_model(
        X_train,
        X_test,
        y_train,
        model_name,
        method_name,
    )

    model = make_model(model_name, method_name)
    model.fit(X_train_used, y_train_used)

    probability = model.predict_proba(X_test_used)[:, 1]
    prediction = (probability >= 0.5).astype(int)

    return {
        "Dataset": dataset_name,
        "Model": model_name,
        "Method": method_name,
        "Threshold": 0.5,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1": f1_score(y_test, prediction, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probability),
        "y_test": y_test,
        "prediction": prediction,
        "probability": probability,
    }


# ----------------------------- Plots -----------------------------

def save_class_distribution_plot(summary):
    fig, ax = plt.subplots(figsize=(10, 6))

    summary.set_index("Dataset")[["Class 0", "Class 1"]].plot(
        kind="bar",
        ax=ax,
        logy=True,
        color=["tab:blue", "tab:orange"],
    )

    ax.set_title("Class Distribution for Both Datasets", fontsize=18)
    ax.set_ylabel("Number of Instances (log scale)", fontsize=13)
    ax.set_xlabel("")
    ax.legend(["Class 0", "Class 1"], loc="upper left")
    plt.xticks(rotation=10, ha="right")

    explanation = (
        "Color / Class Meaning\n"
        "────────────────────\n"
        "Blue = Class 0\n"
        "Orange = Class 1\n\n"
        "Credit Card Fraud\n"
        "Class 0 = Legitimate\n"
        "Class 1 = Fraud\n\n"
        "Breast Cancer Wisconsin\n"
        "Class 0 = Malignant\n"
        "Class 1 = Benign"
    )

    ax.text(
        0.98,
        0.97,
        explanation,
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="white",
            edgecolor="black",
            alpha=0.92,
        ),
    )

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=300)
    plt.show()


def save_f1_plots(metrics):
    for dataset_name in metrics["Dataset"].unique():
        subset = metrics[metrics["Dataset"] == dataset_name].sort_values("F1")
        labels = subset["Model"].str.replace("Logistic Regression", "LR")
        labels = labels + " + " + subset["Method"]
        colors = [METHOD_COLORS[method] for method in subset["Method"]]

        plt.figure(figsize=(8.5, 4.8))
        plt.barh(labels, subset["F1"], color=colors)
        plt.title(f"F1 Scores - {dataset_name}")
        plt.xlabel("F1 score")
        plt.xlim(max(0, subset["F1"].min() - 0.02), 1.0)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color)
            for color in METHOD_COLORS.values()
        ]
        plt.legend(
            handles,
            METHOD_COLORS.keys(),
            title="Method color",
            loc="lower right",
            fontsize=8,
            title_fontsize=8,
        )

        plt.tight_layout()
        filename = dataset_name.lower().replace(" ", "_") + "_f1_scores.png"
        plt.savefig(FIGURE_DIR / filename, dpi=300)
        plt.show()


def save_roc_plots(metrics, results):
    for dataset_name in metrics["Dataset"].unique():
        plt.figure(figsize=(6.8, 5))

        dataset_results = [row for row in results if row["Dataset"] == dataset_name]
        top_rows = sorted(dataset_results, key=lambda row: row["ROC_AUC"], reverse=True)[:4]

        for row in top_rows:
            fpr, tpr, _ = roc_curve(row["y_test"], row["probability"])
            label = f"{row['Model']} + {row['Method']} (AUC={row['ROC_AUC']:.4f})"
            plt.plot(fpr, tpr, label=label)

        plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        plt.title(f"Top 4 ROC Curves by AUC - {dataset_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(fontsize=7)
        plt.tight_layout()

        filename = dataset_name.lower().replace(" ", "_") + "_roc.png"
        plt.savefig(FIGURE_DIR / filename, dpi=300)
        plt.show()


def save_confusion_matrices(metrics, results):
    for dataset_name in metrics["Dataset"].unique():
        best_idx = metrics[metrics["Dataset"] == dataset_name]["F1"].idxmax()
        best_model = metrics.loc[best_idx, "Model"]
        best_method = metrics.loc[best_idx, "Method"]

        best_result = next(
            row
            for row in results
            if row["Dataset"] == dataset_name
            and row["Model"] == best_model
            and row["Method"] == best_method
        )

        cm = confusion_matrix(best_result["y_test"], best_result["prediction"])
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(values_format="d", colorbar=False, cmap="Blues")

        if dataset_name == "Credit Card Fraud":
            note = "Class 0 = Legitimate, Class 1 = Fraud"
        else:
            note = "Class 0 = Malignant, Class 1 = Benign"

        title = f"Best F1 Confusion Matrix - {dataset_name}\n"
        title += f"{best_model} + {best_method}\n"
        title += note

        plt.title(title)
        plt.xlabel(f"Predicted Label\n{note}")
        plt.ylabel(f"True Label\n{note}")
        plt.tight_layout()

        filename = dataset_name.lower().replace(" ", "_") + "_confusion_matrix.png"
        plt.savefig(FIGURE_DIR / filename, dpi=300)
        plt.show()


# ----------------------------- Experiment -----------------------------

def run_experiment():
    X_credit, y_credit = load_credit_card_data()
    X_breast, y_breast = load_breast_cancer_data()

    print("Credit card shape:", X_credit.shape)
    print("Breast cancer shape:", X_breast.shape)

    summary = pd.DataFrame(
        [
            class_summary(y_credit, "Credit Card Fraud"),
            class_summary(y_breast, "Breast Cancer Wisconsin"),
        ]
    )

    print("\nClass summary:")
    print(summary.to_string(index=False))
    save_class_distribution_plot(summary)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_credit,
        y_credit,
        test_size=0.25,
        stratify=y_credit,
        random_state=RANDOM_STATE,
    )

    Xc_train, yc_train = sample_credit_training_set(Xc_train, yc_train)

    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X_breast,
        y_breast,
        test_size=0.25,
        stratify=y_breast,
        random_state=RANDOM_STATE,
    )

    print("\nTraining set after runtime sampling:")
    print("Credit Card Fraud:", dict(pd.Series(yc_train).value_counts().sort_index()))
    print("Breast Cancer Wisconsin:", dict(pd.Series(yb_train).value_counts().sort_index()))

    datasets = [
        ("Credit Card Fraud", Xc_train, Xc_test, yc_train, yc_test),
        ("Breast Cancer Wisconsin", Xb_train, Xb_test, yb_train, yb_test),
    ]

    models = ["Logistic Regression", "Random Forest"]
    methods = [
        "Baseline",
        "Random Oversampling",
        "Random Undersampling",
        "SMOTE",
        "Class Weight",
    ]

    results = []

    for dataset_name, X_train, X_test, y_train, y_test in datasets:
        for model_name in models:
            for method_name in methods:
                print(f"Running: {dataset_name} | {model_name} | {method_name}")
                result = evaluate_model(
                    dataset_name,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    model_name,
                    method_name,
                )
                results.append(result)

    metrics = pd.DataFrame(
        [
            {
                key: value
                for key, value in result.items()
                if key not in ["y_test", "prediction", "probability"]
            }
            for result in results
        ]
    )

    metrics = metrics.sort_values(["Dataset", "F1"], ascending=[True, False])
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)

    print("\nFinal metrics:")
    print(metrics.round(4).to_string(index=False))

    save_f1_plots(metrics)
    save_roc_plots(metrics, results)
    save_confusion_matrices(metrics, results)

    print("\nSaved results to results/metrics.csv")
    print("Saved figures to the figures folder")


if __name__ == "__main__":
    run_experiment()
