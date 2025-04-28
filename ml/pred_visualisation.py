import os
import dotenv

import torch
import numpy as np
import matplotlib.pyplot as plt


dotenv.load_dotenv()#os.path.abspath(os.path.join(os.path.dirname(__file__)), "..", ".env"))
plots_path = os.getenv("PLOT_DIR")

def sequence_pred_plot(model, dataset, index, model_name, unorm_func=None):
    #get gt control values and make model predictions
    images, gts = dataset[index]
    images = images.reshape(dataset.seq_length, 3, images.shape[1], images.shape[2])
    gts = torch.flip(gts, dims=(0,))
    
    model.eval()
    with torch.no_grad():
        preds = model(images)
        if unorm_func: preds = unorm_func(preds)
    preds = torch.flip(preds, dims=(0,))
    
    control_names = dataset.cols_to_predict
    
    session_name = dataset.df.iloc[index]["session"]
    final_frame = int(dataset.df.iloc[index]["frame"])
    first_frame = final_frame - dataset.seq_length
    
    x_axis = np.arange(start=0, stop=dataset.seq_length)
    
    for i in range(gts.shape[1]):
        fig, ax = plt.subplots()
        #plot gt and pred control values
        ax.plot(x_axis, preds[:, i], c = "blue", label="pred")
        ax.plot(x_axis, gts[:, i], c = "green", label="gt") 
        
        #style plot
        ax.set_title(f"{control_names[i]} predictions on session {session_name} (frame {first_frame} - {final_frame})")
        ax.legend()
        ax.set_xticks(x_axis)
        ax.grid(axis='x', alpha=0.7)
        ax.set_xlabel("Frame")
        ax.set_ylabel(f"{control_names[i]} Value")

        os.makedirs(f"{plots_path}/seq_{session_name}/{model_name}", exist_ok=True)
        fig.savefig(f"{plots_path}/seq_{session_name}/{model_name}/s{session_name}_f{first_frame}_f{final_frame}_{control_names[i]}.png")
        

if __name__ == "__main__":
    import torchvision.transforms.v2 as v2
    
    from model_utils import load_model
    from models import EfficientnetAgent
    from dataset import VRNET2_Single_Session_Seq_Controls_Dataset, ORIGNAL_COLS_TO_PREDICT
    
    im_size = (456, 456)
    model = EfficientnetAgent(im_size=im_size, output_size=10, size=4, pretrained=False)
    load_model(model, "runs/ENB4_MSE_normed/weights/Epoch_5.pt")
    print("Loaded Model")
    
    dataset = VRNET2_Single_Session_Seq_Controls_Dataset("14_2", 30, cols_to_predict=ORIGNAL_COLS_TO_PREDICT, transform=v2.Resize(im_size))
    print("Loaded Dataset\n")
    
    
    def unorm(preds):
        batch_size = preds.shape[0]
        unorm_factor = torch.tensor([3, 3, 3, 10, 10, 10, 1, 1, 1, 1])
        unorm_factor = torch.stack([unorm_factor for i in range(batch_size)])
        
        preds *= unorm_factor
        return preds
    
    sequence_pred_plot(model, dataset, 6336, "ENB4_MSE_normed", unorm_func=unorm)