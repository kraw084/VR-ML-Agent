import os

import dotenv
import pandas as pd
from tqdm import tqdm

dotenv.load_dotenv()
dataset_path = os.getenv("DATASET_PATH")


for session in os.listdir(dataset_path):
    #read csv
    print(f"Formatting {session} data")
    first_num = session.split("_")[0]
    csv = pd.read_csv(f"{dataset_path}/{session}/{first_num}_data.csv")
    csv_length = len(csv)
    
    #remove uneeded columns
    cols_to_remove = ["timestamp", "object_name", "mesh_name", "bounds", "m_matrix", "object_id", "mesh_id",
                  "static", "camera_name", "p_matrix", "v_matrix", "camera_id", "is_main", 
                  "ConnectedControllerTypes", "ControllerMask"]
    cols_to_remove += [c for c in csv.columns if "acc" in c]    
    csv = csv.drop(columns = cols_to_remove)

    #rename a column and add a session column
    csv = csv.rename(columns={"key":"frame"})
    csv.insert(0, "session", [session] * len(csv))
    
    #find all frame numbers for this session
    im_names = os.listdir(f"{dataset_path}/{session}/video")
    im_names = [int(n[:-4]) for n in im_names]
    
    #remove rows with no images
    bool_mask = (csv["frame"].isin(im_names))
    csv = csv[bool_mask]
            
    print(f"Removed {csv_length - bool_mask.sum()} rows")
    print("Finished formatting")
    
    #save as a new csv
    csv.to_csv(f"{dataset_path}/{session}/{session}_data.csv")
    
    print()