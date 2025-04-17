import os

import dotenv
import torch
import torchvision
import pandas as pd

dotenv.load_dotenv()
dataset_path = os.getenv("DATASET_PATH")

DEFAULT_COLS_TO_PREDICT = ['head_vel_x', 'head_vel_y', 'head_vel_z', 'head_angvel_x', 'head_angvel_y', 'head_angvel_z',
                           'left_controller_vel_x', 'left_controller_vel_y', 'left_controller_vel_z',
                           'left_controller_angvel_x', 'left_controller_angvel_y', 'left_controller_angvel_z', 
                           'right_controller_vel_x', 'right_controller_vel_y', 'right_controller_vel_z', 
                           'right_controller_angvel_x', 'right_controller_angvel_y', 'right_controller_angvel_z',
                           'Thumbstick_0_x', 'Thumbstick_0_y', 'Thumbstick_1_x', 'Thumbstick_1_y']

ORIGNAL_COLS_TO_PREDICT = ['head_vel_x', 'head_vel_y', 'head_vel_z', 'head_angvel_x', 'head_angvel_y', 'head_angvel_z',
                           'Thumbstick_0_x', 'Thumbstick_0_y', 'Thumbstick_1_x', 'Thumbstick_1_y']


def read_sessions_txt(name):
    with open(f"VR-ML-Agent/datasets/{name}.txt") as f:
        sessions = f.read().strip("\n").split("\n")
        
    return sessions


def nan_count(df):
    return df.isna().sum().sum().item()


class VRNET2_Dataset_Template(torch.utils.data.Dataset):
    def __init__(self, seq_length=1, cols_to_predict=None, transform=None, target_transform=None):
        """A default template for VRNET2.0 datasets, subclass must extend init to create the df in the desired format"""
        self.seq_length = seq_length
        self.cols_to_predict = DEFAULT_COLS_TO_PREDICT if cols_to_predict is None else cols_to_predict
        self.transform = transform
        self.target_transform = target_transform
        
        self.df = None
        
    def prep_session_csv(self, session_name):
        #read the csv and remove uneeded rows
        df = pd.read_csv(f"{dataset_path}/{session_name}/{session_name}_data.csv")
        df = df[["session", "frame"] + self.cols_to_predict]
        
        #remove beginning rows so images with not enough previous frames cannot be selected
        if self.seq_length > 1:
            self.df = self.df.drop([i for i in range(0, self.seq_length)])
            
        #num_of_nan_vals = nan_count(df)
        #if num_of_nan_vals > 0:
        #    raise ValueError(f"WARNING: dataset contains {num_of_nan_vals} NaN values")
         
        df = df.dropna()
            
        return df
        
    def __len__(self):
        return len(self.df)
    
    def get_im_path(self, session_name, frame):
        return f"{dataset_path}/{session_name}/video/{frame}.jpg"
    
    def __getitem__(self, index):
        data_row = self.df.iloc[index]
        
        if self.seq_length == 1:
            #read the single frame and turn it into a tensor
            im_path = self.get_im_path(data_row["session"], data_row["frame"])
            x = (torchvision.io.read_image(im_path) / 255).float()
        else:
            session = data_row["session"]
            final_frame = int(data_row["frame"])
            x = []
            
            #read each frame in the sequence and convert to a tensor
            for i in range(0, self.seq_length):
                x.append((torchvision.io.read_image(im_path(session, final_frame - i))/255).float())
            x = torch.concat(x)
            
        #extract prediction target
        y = data_row[self.cols_to_predict]
        y = torch.from_numpy(y.to_numpy(dtype=float)).float()
        
        #apply transformations
        if self.transform: x = self.transform(x)
        if self.target_transform: y = self.target_transform(y)
        
        return x, y


class VRNET2_Single_Session_Dataset(VRNET2_Dataset_Template):
    def __init__(self, session, seq_length=1, cols_to_predict=None, transform=None, target_transform=None):
        super().__init__(seq_length, cols_to_predict, transform, target_transform)
        
        self.session = session
        self.df = self.prep_session_csv(session)
        
        
        
        
class VRNET2_Multi_Session_Dataset(VRNET2_Dataset_Template):
    def __init__(self, sessions, seq_length=1, cols_to_predict=None, transform=None, target_transform=None):
        super().__init__(seq_length, cols_to_predict, transform, target_transform)
        
        if type(sessions) is str: sessions = read_sessions_txt(sessions)
        
        self.sessions = sessions
        self.df = pd.concat([self.prep_session_csv(s) for s in self.sessions])
    