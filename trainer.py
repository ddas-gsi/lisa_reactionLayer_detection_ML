# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import pandas as pd

# from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE

# def train_hgb_with_smote_cv_save(
#     X_train, y_train, X_test, y_test,
#     class_weight=None,
#     model_path="hgb_model.joblib",
#     random_state=42,
#     do_cv=True,
#     cv_folds=5
# ):
#     """
#     Trains a HistGradientBoostingClassifier with SMOTE, cross-validation, and model saving.

#     Parameters:
#         X_train, y_train: Training data
#         X_test, y_test: Test data
#         class_weight: Dictionary for class weights (e.g., {2.0: 2})
#         model_path: File path to save the model
#         random_state: Random seed
#         do_cv: Whether to perform cross-validation
#         cv_folds: Number of folds for CV

#     Returns:
#         model: Trained classifier
#     """

#     # 🧪 SMOTE Oversampling
#     print("🔁 Applying SMOTE to training data...")
#     smote = SMOTE(random_state=random_state)
#     X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
#     print(f"✅ Resampled training shape: {X_train_bal.shape}, {y_train_bal.shape}")

#     # 📊 Optional Cross-validation
#     if do_cv:
#         print(f"🔍 Running {cv_folds}-Fold Cross-Validation...")
#         temp_model = HistGradientBoostingClassifier(random_state=random_state, class_weight=class_weight)
#         scores = cross_val_score(temp_model, X_train_bal, y_train_bal, cv=cv_folds, scoring='accuracy')
#         print(f"✅ CV Mean Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

#     # 🧠 Training
#     print("🚀 Training HistGradientBoostingClassifier...")
#     clf = HistGradientBoostingClassifier(random_state=random_state, class_weight=class_weight)
#     clf.fit(X_train_bal, y_train_bal)

#     # 🔎 Evaluation on test set
#     y_pred = clf.predict(X_test)
#     print("\n📈 Confusion Matrix:")
#     cm = confusion_matrix(y_test, y_pred)
#     print(cm)

#     print("\n📋 Classification Report:")
#     print(classification_report(y_test, y_pred))

#     # 📊 Plot confusion matrix
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=clf.classes_, yticklabels=clf.classes_)
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.show()

#     # 💾 Save the model
#     joblib.dump(clf, model_path)
#     print(f"\n💾 Model saved to {model_path}")

#     return clf


# if __name__ == "__main__":

#     df = pd.read_csv("Pareeksha_data_2.csv")
#     df["dE_Tot"] = df["dE_L0"] + df["dE_L1"]

#     # Feature and target selection
#     X = df[["dE_L0", "dE_L1", "dE_Tot"]]
#     # X = df[["dE_L0", "dE_L1"]]
#     y = df["ReactionLayerID"]

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model_hgb = train_hgb_with_smote_cv_save(
#         X_train, y_train, X_test, y_test,
#         class_weight={2.0: 2},
#         model_path="./models/hgb_model_smote.joblib",
#         do_cv=True
#     )


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
