import dotenv
import os
import pandas as pd

dotenv.load_dotenv()
dataset_path = os.getenv("DATASET_PATH")

csv = pd.read_csv(f"{dataset_path}/15_2/15_data.csv")

print(csv.columns)