import os
import zipfile
import shutil

from tqdm import tqdm
import cv2
import dotenv

#setup env variables
dotenv.load_dotenv()

downloaded_dataset_path = os.getenv("DOWNLOADED_DATASET_PATH")
new_dataset_path = os.getenv("DATASET_PATH")
if not os.path.exists(new_dataset_path): os.mkdir(new_dataset_path)

#counters for controlling which sessions to start/stop with
counter = 0
start_counter = 0
end_counter = 1000

print("All Sessions:")
print(os.listdir(downloaded_dataset_path))

for session in os.listdir(downloaded_dataset_path):
    #counter logic to skip sessions or end loop
    if counter < start_counter:
        counter += 1
        continue
    elif counter >= end_counter:
        break
    else:
        counter += 1
    
    print(f"Extracting {session.split('_')[0]}_video.zip")
    
    identifier = session.split("_")[0]
    original_session_folder = f"{downloaded_dataset_path}/{session}/"
    new_session_folder = f"{new_dataset_path}/{session}/"
    
    #create session folder in new dataset
    if not os.path.exists(f"{new_session_folder}"): os.mkdir(f"{new_session_folder}")
    
    #extract video zip
    with zipfile.ZipFile(f"{original_session_folder}/{identifier}_video.zip","r") as zip_file:
        zip_file.extractall(f"{new_session_folder}") 
        
    #remove extra file if it is in the zip
    if os.path.exists(f"{new_dataset_path}/{session}/__MACOSX"): shutil.rmtree(f"{new_dataset_path}/{session}/__MACOSX")
        
    #if images were not bundled in a video folder, create one
    if not os.path.exists(f"{new_session_folder}/video"):
        os.mkdir(f"{new_session_folder}/video")
        for im_name in os.listdir(f"{new_session_folder}"):
            if im_name.endswith(".jpg"):
                os.rename(f"{new_session_folder}/{im_name}", f"{new_session_folder}/video/{im_name}")
        
    print("Finished unzipping")
    
    #shrink all images by half and rename them
    for im_path in tqdm(os.listdir(f"{new_session_folder}/video"), desc="Resizing images"):
       im = cv2.imread(f"{new_session_folder}/video/{im_path}")
       im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
       cv2.imwrite(f"{new_session_folder}/video/{im_path[1:]}", im)
       os.remove(f"{new_session_folder}/video/{im_path}") 
    
    #delete the original zip
    os.remove(f"{original_session_folder}/{identifier}_video.zip")
    print("Removed zip")
    
    print()