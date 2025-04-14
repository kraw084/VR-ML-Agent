import os
import zipfile
import shutil

from tqdm import tqdm
import cv2
import dotenv

dotenv.load_dotenv()
dataset_path = os.getenv("DATASET_PATH")

new_dataset_path = os.path.abspath(os.path.join(dataset_path, "..", "VRNET2.0_Unpacked"))
if not os.path.exists(new_dataset_path): os.mkdir(new_dataset_path)

for session in os.listdir(dataset_path):
    print(f"Extracting {session.split('_')[0]}_video.zip")
    
    identifier = session.split("_")[0]
    original_session_folder = f"{dataset_path}/{session}/"
    new_session_folder = f"{new_dataset_path}/{session}/"
    
    if not os.path.exists(f"{new_session_folder}"): os.mkdir(f"{new_session_folder}")
        
    with zipfile.ZipFile(f"{original_session_folder}/{identifier}_video.zip","r") as zip_file:
        zip_file.extractall(f"{new_session_folder}") 
        
    shutil.rmtree(f"{new_dataset_path}/{session}/__MACOSX")
        
    print("Finished unzipping")
    
    for im_path in tqdm(os.listdir(f"{new_session_folder}/video"), desc="Resizing images"):
       im = cv2.imread(f"{new_session_folder}/video/{im_path}")
       im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
       cv2.imwrite(f"{new_session_folder}/video/{im_path[1:]}", im)
       os.remove(f"{new_session_folder}/video/{im_path}") 
    
    print()