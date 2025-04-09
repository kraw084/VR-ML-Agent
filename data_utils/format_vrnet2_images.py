import os
import zipfile
import shutil

from tqdm import tqdm
import cv2
import dotenv

dotenv.load_dotenv()
dataset_path = os.getenv("DATASET_PATH")

for session in os.listdir(dataset_path):
    print(f"Extracting {session.split('_')[0]}_video.zip")
    
    identifier = session.split("_")[0]
    session_folder = f"{dataset_path}/{session}/"
        
    with zipfile.ZipFile(f"{session_folder}/{identifier}_video.zip","r") as zip_file:
        zip_file.extractall(f"{session_folder}") 
        
    shutil.rmtree(f"{session_folder}/__MACOSX")
        
    print("Finished unzipping")
    
    for im_path in tqdm(os.listdir(f"{session_folder}/video"), desc="Resizing images"):
       im = cv2.imread(f"{session_folder}/video/{im_path}")
       im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
       cv2.imwrite(f"{session_folder}/video/{im_path[1:]}", im)
       os.remove(f"{session_folder}/video/{im_path}") 
    
    print()