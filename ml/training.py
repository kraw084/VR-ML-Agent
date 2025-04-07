import os

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import dotenv

from model_utils import load_model, save_model, evaluate_model


dotenv.load_dotenv()
model_save_dir = os.getenv("MODEL_SAVE_DIR")

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, epochs, batch_size, lr, loss_fn,
                 weight_decay, save_freq, device, name, workers=1, resume=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.weight_decay = weight_decay
        self.save_freq = save_freq
        self.device = device
        self.name = name
        self.resume = resume
        self.workers = workers

        if resume:
            #load model is resume is specified
            load_model(self.model, f"{model_save_dir}/{name}/weights/Epoch_{self.resume}.pt")
        else:
            #otherwise setup files if they dont already exist 
            if not os.path.exists(f"{self.save_path}"): os.mkdir(f"{self.save_path}")
            if not os.path.exists(f"{self.save_path}/{self.name}"): os.mkdir(f"{self.save_path}/{self.name}")
            if not os.path.exists(f"{self.save_path}/{self.name}/weights"): os.mkdir(f"{self.save_path}/{self.name}/weights")
            if not os.path.exists(f"{self.save_path}/{self.name}/logs"): os.mkdir(f"{self.save_path}/{self.name}/logs")
            
            #save hyperparameters
            with open(f"{self.save_path}/{self.name}/hyps.txt", "w") as f:
                f.write(f"Model Name: {self.name}\n")
                f.write(f"Epochs: {self.epochs}\n")
                f.write(f"Batch Size: {self.batch_size}\n")
                f.write(f"Learning Rate: {self.lr}\n")
                f.write(f"Weight Decay: {self.weight_decay}\n")
                f.write(f"Save Frequency: {self.save_freq}\n")
                f.write(f"Device: {self.device}\n")


        #create data loader and optimizer
        self.loader = torch.utils.data.DataLoader(train_dataset, batch_size, workers=self.workers)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        #create tensorboard
        self.writer = SummaryWriter(f"{self.save_path}/{self.name}/logs")


    def process_batch(self, e_i, batch_X, batch_Y, progress_bar):
        """Handles calculating loss for a single batch"""
        #run the batch through the model and calculate loss
        prediction = self.model(batch_X)
        loss = self.loss_fn(prediction, batch_Y)

        #optimization step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        batch_loss = loss.item()

        #append loss to progress bar
        progress_bar.set_postfix({"loss": batch_loss})

        #add batch loss to tensorboard
        self.writer.add_scalar('Loss/Batch_Loss', batch_loss, e_i * len(self.loader) + progress_bar.n)

        return batch_loss
    

    def process_epoch(self, e_i):
        """Handles training for a single epoch"""
        self.model.train()
        
        #create progress bar for the epoch
        progress_bar = tqdm(self.loader, desc = f"Epoch {e_i}/{self.epochs}", bar_format = "{l_bar}{bar:20}{r_bar}")

        #loop over all batchs in the epoch and perform the training step
        epoch_loss = 0
        for batch in progress_bar:
            batch_loss = self.process_batch(e_i, batch, progress_bar)
            epoch_loss += batch_loss

        #calculate average epoch loss and add it to the tensorboard
        avg_epoch_loss = epoch_loss / len(self.loader)
        self.writer.add_scalar('Loss/Train_Loss', avg_epoch_loss, e_i)
        print(f"Epoch {e_i} finished - Avg Train loss: {avg_epoch_loss}")

        #save model weights
        if (e_i % self.save_freq) == 0 or e_i == self.epochs - 1:
            save_model(self.model, f"{model_save_dir}/{self.name}/weights/Epoch_{e_i}.pt")
            
            #validate model and add val loss to tensorboard
            eval_loss = evaluate_model(self.model, self.val_dataset, self.batch_size, self.loss_fn, self.device, self.workers)
            self.writer.add_scalar('Loss/Val_Loss', eval_loss, e_i)
            print(f"Epoch {e_i} validation loss: {eval_loss}")

        print()


    def train(self):
        """Main training loop, call to initiate model training"""
        #loop over epochs
        for e_i in range(0 if self.resume is None else self.resume + 1, self.epochs):
            self.process_epoch(e_i)

        self.writer.close()
        print("Training finished!")