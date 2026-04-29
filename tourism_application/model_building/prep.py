import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# DATA_DIR is not used here as data is directly loaded or downloaded from HF
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load data from HF or local
try:
    # Attempt to load locally first (e.g., if already downloaded by data_register.py)
    df = pd.read_csv("tourism_application/data/tourism.csv")
except FileNotFoundError:
    # If not found locally, download from Hugging Face Space
    print("tourism.csv not found locally, downloading from Hugging Face.")
    df = pd.read_csv(f"https://huggingface.co/spaces/vyasmax9/tourism-predict-app/resolve/main/data/tourism.csv")

df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True, errors="ignore")

# No preprocessing (like LabelEncoder) here. Pass raw data to train.py

# Split
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save splits
splits_dir = Path("splits")
splits_dir.mkdir(exist_ok=True)
for name, data in [("X_train", X_train), ("X_test", X_test), ("y_train", y_train), ("y_test", y_test)]:
    data.to_csv(splits_dir / f"{name}.csv", index=False)

# Upload splits
api.upload_folder(
    folder_path=str(splits_dir),
    path_in_repo="data/splits",
    repo_id="vyasmax9/tourism-predict-app",
    repo_type="space"
)
print("✅ Data prep complete!")
