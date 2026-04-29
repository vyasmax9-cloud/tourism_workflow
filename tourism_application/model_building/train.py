import os
import joblib
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, recall_score
from huggingface_hub import HfApi
import mlflow

mlflow.set_experiment("tourism-prediction-data-scientist")

api = HfApi(token=os.getenv("HF_TOKEN"))

# Ensure the 'splits' directory exists for local loading
splits_dir = Path("splits")
splits_dir.mkdir(exist_ok=True) # Create if it doesn't exist

# Download splits from Hugging Face if not present locally
# This assumes data_prep already uploaded them to data/splits
from huggingface_hub import hf_hub_download
for filename in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    if not (splits_dir / filename).exists():
        print(f"Downloading {filename} from Hugging Face.")
        hf_hub_download(
            repo_id="vyasmax9/tourism-predict-app",
            filename=f"data/splits/{filename}", # Note the subfolder path
            local_dir=splits_dir,
            repo_type="space"
        )


# Load splits
X_train = pd.read_csv(splits_dir / "X_train.csv")
y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze()
X_test = pd.read_csv(splits_dir / "X_test.csv")
y_test = pd.read_csv(splits_dir / "y_test.csv").squeeze()

# Define preprocessor
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Pipeline with RandomForestClassifier
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model_pipeline, param_grid=param_grid, cv=3, scoring='recall', n_jobs=-1, verbose=1)

with mlflow.start_run():
    print("Starting GridSearchCV for hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    print("GridSearchCV complete. Best parameters logged.")

    # Evaluate & log for best model
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)

    # Log detailed metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_recall_0": train_report['0']['recall'],
        "train_precision_0": train_report['0']['precision'],
        "train_f1_0": train_report['0']['f1-score'],
        "train_recall_1": train_report['1']['recall'],
        "train_precision_1": train_report['1']['precision'],
        "train_f1_1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_recall_0": test_report['0']['recall'],
        "test_precision_0": test_report['0']['precision'],
        "test_f1_0": test_report['0']['f1-score'],
        "test_recall_1": test_report['1']['recall'],
        "test_precision_1": test_report['1']['precision'],
        "test_f1_1": test_report['1']['f1-score'],
        "best_cv_score": grid_search.best_score_
    })
    print("Detailed metrics logged to MLflow.")

    # Save locally
    joblib.dump(best_model, "model.pkl")
    mlflow.sklearn.log_model(best_model, "tourism_model")
    print("Model saved locally and logged as MLflow artifact.")

# Upload best model to Hugging Face
api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id="vyasmax9/tourism-predict-app",
    repo_type="space"
)
print(f"✅ Model trained and uploaded! Test recall (class 1): {test_report['1']['recall']:.3f}")
