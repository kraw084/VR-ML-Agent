import os
import zipfile
import cv2

import dotenv

dotenv.load_dotenv()
dataset_path = os.getenv("DATASET_PATH")

for session in os.listdir(dataset_path):
    print(f"Extracting {session.split("_")[0]}_video.zip")
    
    if not os.path.exists(f"{dataset_path}/frames"): os.mkdir(f"{dataset_path}/frames")
    
    with zipfile.ZipFile(f"{dataset_path}/{session.split("_")[0]}_video.zip","r") as zip_ref:
        zip_ref.extractall(f"{dataset_path}/frames") #TEST WHERE THIS GOES
        
    print("Finished unzipping")
    
    for im_path in os.listdir(f"{dataset_path}/frames"):
       pass 
        
    break