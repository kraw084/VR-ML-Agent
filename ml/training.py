import os
import re

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import dotenv

from ml.model_utils import load_model, save_model, evaluate_model


dotenv.load_dotenv()
model_save_dir = os.getenv("MODEL_SAVE_DIR")

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, epochs, batch_size, lr, loss_fn,
                 weight_decay, save_freq, device, name, workers=1, resume=None, scheduler=None):
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
            if not os.path.exists(f"{model_save_dir}"): os.mkdir(f"{model_save_dir}")

            if not os.path.exists(f"{model_save_dir}/{self.name}"): 
                os.mkdir(f"{model_save_dir}/{self.name}")
            else:
                pattern = re.compile(rf"^{re.escape(name)}(_\d+)?$")
                numbers = []
                for n in os.listdir(model_save_dir):
                    re_match = pattern.fullmatch(n)
                    if re_match: numbers.append(int(re_match.group(1)[1:]) if re_match.group(1) else 0)
                    
                self.name = name + f"_{max(numbers) + 1}"
                os.mkdir(f"{model_save_dir}/{self.name}")
                
            os.mkdir(f"{model_save_dir}/{self.name}/weights")
            os.mkdir(f"{model_save_dir}/{self.name}/logs")
            
            #save hyperparameters
            with open(f"{model_save_dir}/{self.name}/hyps.txt", "w") as f:
                f.write(f"Model Name: {self.name}\n")
                f.write(f"Model Type: {str(self.model)}\n")
                f.write(f"Epochs: {self.epochs}\n")
                f.write(f"Batch Size: {self.batch_size}\n")
                f.write(f"Learning Rate: {self.lr}\n")
                f.write(f"Weight Decay: {self.weight_decay}\n")
                f.write(f"Save Frequency: {self.save_freq}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"Cols to predict: {self.train_dataset.cols_to_predict}\n")
                
                if not scheduler is None:
                    f.write(f"Scheduler: {str(scheduler).split(' ')[1]}\n")
                
                f.write(f"\nNotes: \n")
                
            
        self.model.to(self.device)

        #create data loader and optimizer
        self.loader = torch.utils.data.DataLoader(train_dataset, batch_size, num_workers=self.workers)
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if not scheduler is None:
            scheduler_class, schedular_kwargs = scheduler
            self.schedular = scheduler_class(self.opt, **schedular_kwargs)
            
        #create tensorboard
        self.writer = SummaryWriter(f"{model_save_dir}/{self.name}/logs")

    def evaluate(self):
        return evaluate_model(self.model, self.val_dataset, self.batch_size, self.loss_fn, self.device, self.workers)


    def process_batch(self, e_i, batch, progress_bar):
        """Handles calculating loss for a single batch"""
        #run the batch through the model and calculate loss
        batch_X, batch_Y = batch
        batch_X, batch_Y = batch_X.to(self.device), batch_Y.to(self.device)
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
            eval_loss = self.evaluate()
            self.writer.add_scalar('Loss/Val_Loss', eval_loss, e_i)
            print(f"Epoch {e_i} validation loss: {eval_loss}")

        print()
        
        if not self.schedular is None:
            self.schedular.step()


    def train(self):
        """Main training loop, call to initiate model training"""
        #loop over epochs
        for e_i in range(0 if self.resume is None else self.resume + 1, self.epochs):
            self.process_epoch(e_i)

        self.writer.close()
        print("Training finished!")
       

               
class EBMTrainer(Trainer):
    def sample_negatives(self, batch_con):
        """
        B, _ = batch.shape
        
        indicies = torch.arange(B).unsqueeze(0).expand(B, B)
        mask = torch.eye(B, dtype=torch.bool)
        indicies = indicies[~mask]
        indicies = indicies.view((B, B - 1))
        rand_indices = torch.randint(0, B - 1, size=(B, num_negatives))
        
        chosen_indicies = indicies.gather(dim=1, index=rand_indices)
        return batch[chosen_indicies]
        """
        return torch.rand_like(batch_con, device=batch_con.device)
   
        
    def generate_fake_energies(self, batch_con, im_features, number_of_negatives):
        fake_engeries = torch.zeros((self.batch_size, number_of_negatives, 1), device=batch_con.device)
        
        for i in range(number_of_negatives):
            batch_fake_con = self.sample_negatives(batch_con)
            batch_fake_con_features = self.model.extract_control_feature(batch_con)
            batch_fake_energies = self.model.head(im_features, batch_fake_con_features)
            
            fake_engeries[:, i, :] = batch_fake_energies
            
        return fake_engeries
        
     
    def evaluate(self):
        self.model.eval()
        dataloader = torch.utils.data.dataloader.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.workers)
        total_loss = 0
        number_of_negatives = 10
            
        #loop over batches and compute the loss
        with torch.no_grad():
            for batch_ims, batch_con in tqdm(dataloader, desc = "Evaluating model", bar_format = "{l_bar}{bar:20}{r_bar}"):
                batch_ims, batch_con = batch_ims.to(self.device), batch_con.to(self.device)
                
                batch_im_features = self.model.extract_image_feature(batch_ims)
                batch_con_features = self.model.extract_control_feature(batch_con)
                batch_energy = self.model.run_head(batch_im_features, batch_con_features)

                batch_fake_energy = self.generate_fake_energies(batch_con, batch_im_features, number_of_negatives)
                
                total_loss += self.loss_fn(batch_energy, batch_fake_energy)
                                    
        return total_loss/len(dataloader)
       
        
    def process_batch(self, e_i, batch, progress_bar):
        """Handles calculating loss for a single batch"""
        
        number_of_negatives = 10
        
        #run the batch through the model and calculate loss
        batch_ims, batch_con = batch
        batch_ims, batch_con = batch_ims.to(self.device), batch_con.to(self.device)
        
        batch_im_features = self.model.extract_image_feature(batch_ims)
        batch_con_features = self.model.extract_control_feature(batch_con)
        batch_energy = self.model.run_head(batch_im_features, batch_con_features)
        
        batch_fake_energy = self.generate_fake_energies(batch_con, batch_im_features, number_of_negatives)
       
        loss = self.loss_fn(batch_energy, batch_fake_energy)
            
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