import os
import dotenv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#setup env vars and folder paths
dotenv.load_dotenv()
dataset_path = os.getenv("DATASET_PATH")
plots_path = os.getenv("PLOT_DIR")

os.makedirs(plots_path, exist_ok=True)

#generator for looping through all csvs in the dataset
def csv_gen():
    for session in os.listdir(dataset_path):
        csv_path = f"{dataset_path}/{session}/{session}_data.csv"
        csv = pd.read_csv(csv_path)
        
        yield csv
    
def session_im_counts_plot(): 
    #get number of images in each session
    im_counts = []
    for csv in csv_gen():
        im_counts.append(len(csv))
    
    #create dotplot
    fig, ax = plt.subplots()
    ax.plot(im_counts, np.zeros_like(im_counts), 'o', markersize=5, fillstyle="none", markeredgecolor="black")
    ax.set_xlabel('Image count')
    ax.set_xticks(np.arange(start=0, stop=60001, step=10000))
    ax.set_yticks([])
    ax.set_title('Image counts across game sessions')
    ax.grid(axis='x', alpha=0.7)

    fig.subplots_adjust(bottom=0.2) 
    fig.text(0.5, 0.05, f"Total number of images/rows: {sum(im_counts)}", fontsize = 9, ha="center")

    fig.savefig(f"{plots_path}/image_counts.png")
    
    
def dataset_hist(col_name):
    #create pd series of all values for the given column
    values = None
    for csv in csv_gen():
        if values is None: 
            values = csv[col_name]
        else:
            values = pd.concat([values, csv[col_name]])
        
    #create histogram
    fig, ax = plt.subplots()
    ax.hist(values, bins=30)
    ax.set_xlabel(f"{col_name} value")
    ax.set_ylabel(f"Frequency")
    ax.set_title(f"{col_name} Histogram")
    
    #add summary statistics
    fig.subplots_adjust(bottom=0.2) 
    text = f"Min: {values.min()} - Max: {values.max()} - Mean: {values.mean()} - Median: {values.median()}"
    fig.text(0.5, 0.05, text, ha="center", fontsize=8)
    
    fig.savefig(f"{plots_path}/{col_name}_hist.png")
        
    
if __name__ == "__main__":
    #session_im_counts_plot()

    for col in (next(csv_gen()).columns)[2:]:
        dataset_hist(col)