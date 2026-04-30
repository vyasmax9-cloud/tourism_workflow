# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "vyasmax9/tourism-predict-app".strip()
print(f"Using Repo ID:'{repo_id}'")
DATASET_PATH = os.path.join(os.getcwd(),"tourism_application", "data","tourism.csv")

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print("Columns:", df.columns.tolist())

# Drop the unique identifier or index column

df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True) # Drop 'Unnamed: 0' and 'CustomerID' as they are unique identifiers

# Encoding the categorical 'Type' column
label_encoder = LabelEncoder()
df['TypeofContact'] = label_encoder.fit_transform(df['TypeofContact'])

target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id=repo_id,
        repo_type="dataset",
    )
