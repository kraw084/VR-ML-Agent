import os
import random
import dotenv

dotenv.load_dotenv()
dataset_path = os.getenv("DATASET_PATH")

#read list of all sessions
with open("VR-ML-Agent/datasets/all_sessions.txt") as f:
    all_sessions = f.read().strip("\n").split("\n")
    
#define partition proportions
train_prop, val_prop = 0.7, 0.15
train_set, val_set, test_set = [], [], []

#shuffle data and determine start and stop indices
random.shuffle(all_sessions)
train_end_index = round(len(all_sessions) * train_prop)
val_end_index = round(len(all_sessions) * (train_prop + val_prop))

#partition the data
train_set = all_sessions[:train_end_index]
val_set = all_sessions[train_end_index:val_end_index]
test_set = all_sessions[val_end_index:]

#get counts of images in each set
im_counts = {s:len(os.listdir(f"{dataset_path}/{s}/video")) for s in all_sessions}
train_set_count = sum([im_counts[s] for s in train_set])
val_set_count = sum([im_counts[s] for s in val_set])
test_set_count = sum([im_counts[s] for s in test_set])
total_count = train_set_count + val_set_count + test_set_count

#print count stats
print(f"Train set - {len(train_set)} sessions - {train_set_count} images ({round(train_set_count/total_count, 3) * 100}%)")
print(f"Val set - {len(val_set)} sessions - {val_set_count} images ({round(val_set_count/total_count, 3) * 100}%)")
print(f"Test set - {len(test_set)} sessions - {test_set_count} images ({round(test_set_count/total_count, 3) * 100}%)")

#save as txts
def save_txt(name, sessions):
    with open(f"VR-ML-Agent/datasets/{name}.txt", "w") as f:
        f.writelines([s + "\n" for s in sessions])
        
save_txt("train", train_set)
save_txt("val", val_set)
save_txt("test", test_set)
