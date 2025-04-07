import dotenv
import os
import pandas as pd

dotenv.load_dotenv()
dataset_path = os.getenv("DATASET_PATH")

csv = pd.read_csv(f"{dataset_path}/15_2/15_data.csv")

cols_to_remove = ["timestamp", "object_name", "mesh_name", "bounds", "m_matrix", "object_id", "mesh_id",
                  "static", "camera_name", "p_matrix", "v_matrix", "camera_id", "is_main", 
                  "ConnectedControllerTypes", "ControllerMask"]
cols_to_remove += [c for c in csv.columns if "acc" in c]

csv = csv.drop(columns = cols_to_remove)

print(csv.columns)
