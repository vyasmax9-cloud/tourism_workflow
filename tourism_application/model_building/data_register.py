import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, RepositoryNotFoundError

# Dynamic data path - downloads if missing
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Download base dataset if not present
DATA_FILE = DATA_DIR / "tourism.csv"
if not DATA_FILE.exists():
    print("Downloading base dataset...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="vyasmax9/tourism-predict-app",
        filename="tourism.csv",
        local_dir=DATA_DIR,
        repo_type="space"
    )

repo_id = "vyasmax9/tourism-predict-app"
repo_type = "space"
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id, repo_type=repo_type)
    print(f"✅ Space '{repo_id}' exists.")
except RepositoryNotFoundError:
    create_repo(repo_id, repo_type=repo_type, private=False)
    print(f"✅ Created space '{repo_id}'.")

api.upload_folder(
    folder_path=str(DATA_DIR),
    path_in_repo="data",
    repo_id=repo_id,
    repo_type=repo_type,
    ignore_patterns=["*.git*"]
)
print("✅ Data uploaded!")
