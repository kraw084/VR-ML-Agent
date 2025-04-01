import os

import pandas as pd
from fastparquet import ParquetFile

#path to the VR.net dataset
VR_NET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../kraw084/VR.net"))

#List of all game names in the VR.net dataset
GAMES_LIST = list(set(["_".join(s.split("_")[2:]) for s in os.listdir(os.path.join(VR_NET_PATH, "videos"))]))


def load_session_data(game_name, player_id, session):
    """Loads a single game session as a pandas dataframe
    Args:
        game_name: name of the game of the target session
        player_id: the players id number
        session: number indicating the session half (1 or 2)
    Return:
        A pandas df containing the data for that session
    """
    file_name = f"{player_id}_{session}_{game_name}.parquet"
    df = ParquetFile(os.path.join(VR_NET_PATH, "parquet", file_name)).to_pandas()
    
    return df


def load_game_data(game_name):
    """Loads all sessions of a single game as a pandas dataframe
    Args:
        game_name: name of the game to load sessions for
    Return:
        A pandas df made by concating all session dfs of a single game
    """
    #find all sessions with the given game name
    all_sessions = os.listdir(os.path.join(VR_NET_PATH, "videos"))
    game_sessions = [s for s in all_sessions if game_name in s]
    
    if not game_sessions: raise ValueError(f"No sessions of {game_name} found")
    
    #load all sessions and concatenate
    session_dfs = [load_session_data(game_name, s.split("_")[0], s.split("_")[1]) for s in game_sessions]
    game_df = pd.concat(session_dfs)
    
    return game_df


def split_multival(df, col_name, new_col_names, split_func):
    """splits a multivalue df column into servel seperate columns
    Args:
        df: df with the column to split
        col_name: multival col to split
        new_col_names: list of names for the new seperated columns
        split_func: function that takes the column and splits it into a dataframe of seperate columns
    Returns:
        The original dataframe with seperate columns for multivalued attribute (in the same location as the orignal column)
    """
    if not col_name in df.columns: return df
    
    #split the multi-value attribute to create a new df
    col_to_split = df[col_name]
    new_cols = split_func(col_to_split)
    new_cols.columns = new_col_names
    
    #insert the new columns into the orignal df
    old_col_index = df.columns.get_loc(col_name)
    for i, new_col in enumerate(new_col_names):
        df.insert(old_col_index + i, new_col, new_cols[new_col])
    
    #remove the multi-value col
    df = df.drop(columns=[col_name])
    
    return df


def format_df(df, seperate_multival = True, cols_to_remove = None):
    """Formats the raw VR.net session data
    Args:
        df: dateframe to format
        seperate_multival: flag to determine if multivalued attributes will be split or left as is
        cols_to_remove: list of column names to remove from the dataset
    Returns:
        The formatted dataframe
    """
    #remove rows with no image
    df = df.dropna(subset=["video"])
    df = df.drop(columns=["video"])
    
    #reorder columns
    df = df[["game_name", "game_session", "frame", "timestamp", "Thumbstick", "IndexTrigger", "HandTrigger",
            "Buttons", "Touches", "NearTouches", "ConnectedControllerTypes", "head_pos", "head_vel", "head_dir",
            "head_angvel", "left_eye_pos", "left_eye_vel", "left_eye_dir", "left_eye_angvel", "right_eye_pos", 
            "right_eye_vel", "right_eye_dir", "right_eye_angvel"]]
    
    if cols_to_remove:
        df = df.drop(columns = cols_to_remove)
    
    #seperate session values into different attributes
    df = split_multival(df, "game_session", ["player_id", "session_num"], lambda x: x.str.split("_", expand=True).loc[:, :1].astype(pd.Int32Dtype))
    
    if seperate_multival:
        #helper functions
        trigger_col_names = lambda name: [name + "_" + c for c in ["left", "right"]]
        pos_col_names = lambda name: [name + "_" + c for c in ["x", "y", "z"]]
        ang_col_names = lambda name: [name + "_" + c for c in ["a", "b", "c", "d"]]
        split_float_tuple = lambda x: x.str.strip("()").str.split(",", expand = True).astype(float)
        
        df = split_multival(df, "Thumbstick", ["thumbstick_left_x", "thumbstick_left_y", "thumbstick_right_x", "thumbstick_right_y"], split_float_tuple)
        df = split_multival(df, "IndexTrigger", trigger_col_names("index_trigger"), split_float_tuple)
        df = split_multival(df, "HandTrigger", trigger_col_names("hand_trigger"), split_float_tuple)
        for part in ["head", "left_eye", "right_eye"]:
            df = split_multival(df, f"{part}_dir", ang_col_names(f"{part}_dir"), split_float_tuple)
            df = split_multival(df, f"{part}_pos", pos_col_names(f"{part}_pos"), split_float_tuple)         
            df = split_multival(df, f"{part}_vel", pos_col_names(f"{part}_vel"), split_float_tuple)
            df = split_multival(df, f"{part}_angvel", pos_col_names(f"{part}_angvel"), split_float_tuple)  
            
    return df

if __name__ == "__main__":
    #data loading demo for a single session
    df = load_session_data("Earth_Gym", 3, 1)
    df = format_df(df)
    print(df.columns)
    print(df.iloc[:5, :])