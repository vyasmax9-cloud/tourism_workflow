from huggingface_hub import HfApi
import os

DATA_PATH = "tourism_application/deployment"

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="DATA_PATH",     # the local folder containing your files
    repo_id="vyasmax9/tourism-predict-app",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
