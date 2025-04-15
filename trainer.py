import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib


def load_and_split_data(df, feature_cols, target_col, test_size=0.2, random_state=42):
    X = df[feature_cols]
    y = df[target_col].astype(float)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_pipeline():
    return ImbPipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("hgb", HistGradientBoostingClassifier(random_state=42))
    ])


def tune_hyperparameters(pipeline, X_train, y_train):
    param_grid = {
        "hgb__learning_rate": [0.05, 0.1],
        "hgb__max_iter": [100, 200],
        "hgb__max_leaf_nodes": [31, 64],
        "hgb__l2_regularization": [0.0, 1.0],
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="f1_macro",
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))


def save_model(model, path="./models/best_hgb_model.joblib"):
    joblib.dump(model, path)
    print(f"💾 Model saved to '{path}'")


def plot_learning_curve(model, X, y, title="Learning Curve"):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, scoring="f1_macro", n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), verbose=0
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.title(title)
    plt.xlabel("Training Size")
    plt.ylabel("F1 Macro Score")
    plt.grid()
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# === 🎯 Main Execution ===
def train_hist_gradient_boosting(df):
    features = ["dE_L0", "dE_L1", "dE_Tot"]
    target = "ReactionLayerID"

    # Load and split
    X_train, X_test, y_train, y_test = load_and_split_data(df, features, target)

    # Create pipeline
    pipeline = create_pipeline()

    # Tune with GridSearchCV
    grid_search = tune_hyperparameters(pipeline, X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print("✅ Best Parameters:", grid_search.best_params_)

    # Evaluate
    evaluate_model(best_model, X_test, y_test)

    # Save model
    save_model(best_model)

    # Plot learning curve
    X_full = pd.concat([X_train, X_test])
    y_full = pd.concat([y_train, y_test])
    plot_learning_curve(best_model, X_full, y_full)


df = pd.read_csv("Pareeksha_data_2.csv")
df["dE_Tot"] = df["dE_L0"] + df["dE_L1"]
train_hist_gradient_boosting(df)
