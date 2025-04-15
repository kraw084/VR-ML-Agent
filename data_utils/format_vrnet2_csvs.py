import os

import dotenv
import pandas as pd


dotenv.load_dotenv()

downloaded_dataset_path = os.getenv("DOWNLOADED_DATASET_PATH")
new_dataset_path = os.getenv("DATASET_PATH")
if not os.path.exists(new_dataset_path): os.mkdir(new_dataset_path)

for session in os.listdir(downloaded_dataset_path):
    #if the images have not be unzipped yet then skip this session
    if not os.path.exists(f"{new_dataset_path}/{session}"): 
        print(f"Skipping {session} - no images found")
        continue
    
    
    #read csv
    print(f"Formatting {session} data")
    first_num = session.split("_")[0]
    csv = pd.read_csv(f"{downloaded_dataset_path}/{session}/{first_num}_data.csv")
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
    im_names = os.listdir(f"{new_dataset_path}/{session}/video")
    im_names = [int(n[:-4]) for n in im_names]
    
    #remove rows with no images
    bool_mask = (csv["frame"].isin(im_names))
    csv = csv[bool_mask]
            
    print(f"Removed {csv_length - bool_mask.sum()} rows")
    print("Finished formatting")
    
    #save as a new csv
    csv.to_csv(f"{new_dataset_path}/{session}/{session}_data.csv", index=False)
    
    print()