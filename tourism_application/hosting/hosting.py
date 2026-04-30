from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Get current working directory
BASE_DIR = os.getcwd()

# Build correct path
DATA_PATH = os.path.join(BASE_DIR, "tourism_application", "deployment")

print("Using path:", DATA_PATH)

# Check if folder exists
if not os.path.isdir(DATA_PATH):
    raise ValueError(f"Folder NOT found: {DATA_PATH}")

# Upload folder
api.upload_folder(
    folder_path=DATA_PATH,
    repo_id="vyasmax9/tourism-predict-app",
    repo_type="space"
)

