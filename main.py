import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from sklearn.impute import SimpleImputer

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH   = "startup_success_dataset.csv"
EXPORT_PATH = "startup_success_model.pkl"
RANDOM_SEED = 42

SUCCESS_STATES   = {"IPO", "Acquisition"}
NUMERIC_COLS     = ["funding_rounds", "founder_experience_years", "team_size",
                    "market_size_billion", "product_traction_users",
                    "burn_rate_million", "revenue_million"]
CATEGORICAL_COLS = ["investor_type", "sector", "founder_background"]

# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(path: str):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at '{path}'. Please check the file path.")
    df["is_success"] = df["outcome"].str.strip().isin(SUCCESS_STATES).astype(int)
    return df


def validate_columns(df: pd.DataFrame):
    required = NUMERIC_COLS + CATEGORICAL_COLS
    missing  = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")


def split_data(df: pd.DataFrame):
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df["is_success"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED
    )
    return X_train.copy(), X_val.copy(), X_test.copy(), y_train, y_val, y_test

# ── Preprocessing ─────────────────────────────────────────────────────────────

def fit_preprocessors(X_train: pd.DataFrame):
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
    scaler      = StandardScaler()

    X_train[NUMERIC_COLS]     = num_imputer.fit_transform(X_train[NUMERIC_COLS])
    X_train[CATEGORICAL_COLS] = cat_imputer.fit_transform(X_train[CATEGORICAL_COLS])
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
    X_train[NUMERIC_COLS]     = scaler.fit_transform(X_train[NUMERIC_COLS])

    return X_train, num_imputer, cat_imputer, scaler


def apply_preprocessors(X: pd.DataFrame, num_imputer, cat_imputer, scaler, reference_columns):
    X = X.copy()
    X[NUMERIC_COLS]     = num_imputer.transform(X[NUMERIC_COLS])
    X[CATEGORICAL_COLS] = cat_imputer.transform(X[CATEGORICAL_COLS])
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
    X = X.reindex(columns=reference_columns, fill_value=0)
    X[NUMERIC_COLS]     = scaler.transform(X[NUMERIC_COLS])
    return X

# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    splits     = {"Train": (X_train, y_train), "Validation": (X_val, y_val), "Test": (X_test, y_test)}
    accuracies = {name: accuracy_score(y, model.predict(X)) for name, (X, y) in splits.items()}

    print("\n" + "=" * 60)
    print("EVALUATION REPORT".center(60))
    print("=" * 60)
    for name, acc in accuracies.items():
        print(f"  {name:<18} {acc * 100:.2f}%")

    y_pred_test = model.predict(X_test)
    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, target_names=["Failed (0)", "Success (1)"]))

    return {
        "y_pred":     y_pred_test,
        "y_prob":     model.predict_proba(X_test)[:, 1],
        "accuracies": accuracies,           # ← fix: thêm vào để generate_plots dùng được
    }

# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(ax, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Predicted Fail", "Predicted Success"],
                yticklabels=["Actual Fail", "Actual Success"])
    ax.set_title("1. Confusion Matrix (Test Set)", fontweight="bold")


def plot_roc_curve(ax, y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc     = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set(xlim=[0, 1], ylim=[0, 1.05],
           xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_title("2. ROC Curve (Test Set)", fontweight="bold")
    ax.legend(loc="lower right")


def plot_feature_weights(ax, model, feature_names):
    df = (pd.DataFrame({"Feature": feature_names, "Weight": model.coef_[0]})
            .sort_values("Weight"))
    top    = pd.concat([df.head(5), df.tail(5)])
    colors = ["green" if w > 0 else "red" for w in top["Weight"]]
    sns.barplot(x="Weight", y="Feature", data=top, palette=colors, ax=ax)
    ax.set_title("3. Top Feature Weights", fontweight="bold")
    ax.set_xlabel("Weight (negative = reduces success, positive = drives success)")
    ax.set_ylabel("Feature")


def plot_precision_recall(ax, y_test, y_prob):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(recall, precision, color="purple", lw=2, label=f"AP = {ap:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision")
    ax.set_title("4. Precision-Recall Curve", fontweight="bold")
    ax.legend()


def plot_accuracy_comparison(ax, accuracies: dict):
    ax.bar(accuracies.keys(), [v * 100 for v in accuracies.values()],
           color=["steelblue", "orange", "green"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("5. Accuracy Comparison", fontweight="bold")
    ax.set_ylim(0, 100)


def generate_plots(model, X_train, y_test, y_pred, y_prob, accuracies: dict):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_confusion_matrix(axes[0, 0], y_test, y_pred)
    plot_roc_curve(axes[0, 1], y_test, y_prob)
    plot_feature_weights(axes[0, 2], model, X_train.columns)
    plot_precision_recall(axes[1, 0], y_test, y_prob)
    plot_accuracy_comparison(axes[1, 1], accuracies)

    axes[1, 2].set_visible(False)  # ô trống cuối

    plt.tight_layout()
    plt.show()

# ── Export ────────────────────────────────────────────────────────────────────

def export_model(model, num_imputer, cat_imputer, scaler, expected_columns):
    bundle = {
        "model":            model,
        "num_imputer":      num_imputer,
        "cat_imputer":      cat_imputer,
        "scaler":           scaler,
        "expected_columns": expected_columns,
    }
    try:
        joblib.dump(bundle, EXPORT_PATH)
        print(f"Model bundle saved to '{EXPORT_PATH}'")
    except Exception as e:
        raise RuntimeError(f"Failed to save model to '{EXPORT_PATH}': {e}") from e

# ── Pipeline ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("Step 1 — Loading data...")
    df = load_data(DATA_PATH)
    validate_columns(df)

    print("Step 2 — Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"  Train / Val / Test: {len(X_train)} / {len(X_val)} / {len(X_test)} samples")

    print("Step 3 — Preprocessing...")
    X_train, num_imputer, cat_imputer, scaler = fit_preprocessors(X_train)
    X_val  = apply_preprocessors(X_val,  num_imputer, cat_imputer, scaler, X_train.columns)
    X_test = apply_preprocessors(X_test, num_imputer, cat_imputer, scaler, X_train.columns)

    print("Step 4 — Training model...")
    model = train_model(X_train, y_train)

    print("Step 5 — Evaluating model...")
    results = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test)

    print("Step 6 — Generating visualisations...")
    generate_plots(model, X_train, y_test, results["y_pred"], results["y_prob"], results["accuracies"])

    print("Step 7 — Exporting model...")
    export_model(model, num_imputer, cat_imputer, scaler, X_train.columns.tolist())


if __name__ == "__main__":
    main()