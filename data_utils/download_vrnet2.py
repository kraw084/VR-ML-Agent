import subprocess
import dotenv
import os

#set server name that stores the VRNET2.0 data
dotenv.load_dotenv()
server_name = os.getenv("SERVER_NAME")

super_folder = "../../mnt/vr2/VRNET2.0/"

try:
    #for each section of the dataset
    for i in range(1, 14):
        if i == 12: continue #there is no section 12
        
        print(f"Downloading data from folder {i}")
        
        #list all folders in the section
        list_cmd = ["ssh", server_name, f"ls -d {super_folder}{i}/*/"]
        result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
        folder_list = result.stdout.strip().split('\n')
        
        #ignore aux_f folder that is in some directories
        if "aux_f" in folder_list: folder_list.remove("aux_f")
        
        #for each session folder in the section
        for session in folder_list:
            identifier = session.split("/")[-2]
            print(f"Downloading {identifier}")
            
            #create folder to copy data into
            if not os.path.exists(f"E:/VRNET2.0/{identifier}"): os.mkdir(f"E:/VRNET2.0/{identifier}")
            
            first_num = identifier.split("_")[0]
            
            #copy the data csv
            if not os.path.exists(f"E:/VRNET2.0/{identifier}/{first_num}_data.csv"):
                download_data_cmd = ["scp", 
                                    f"{server_name}:/mnt/vr2/VRNET2.0/{i}/{identifier}/{first_num}_data.csv", 
                                    f"E:/VRNET2.0/{identifier}/"]
                subprocess.run(download_data_cmd, check=True)
            else:
                print(f"Data already downloaded for {identifier}")

            #copy the video zip
            if not os.path.exists(f"E:/VRNET2.0/{identifier}/{first_num}_video.zip"):
                download_vid_cmd = ["scp", 
                                    f"{server_name}:/mnt/vr2/VRNET2.0/{i}/{identifier}/{first_num}_video.zip", 
                                    f"E:/VRNET2.0/{identifier}/"]
                subprocess.run(download_vid_cmd, check=True)
            else:
                print(f"Video already downloaded for {identifier}")
                
            print()
        
        print()

except subprocess.CalledProcessError as e:
    print(e)