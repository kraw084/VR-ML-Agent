import torch
from tqdm import tqdm

def save_model(model, path):
    """Save model weights to path"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Load model weights from path"""
    model.load_state_dict(torch.load(path, weights_only=True))
    
    
def evaluate_model(model, dataset, batch_size, loss_fn, device, workers=1):
    """
    Evaluate the model on the given (validation or test) dataset
    Args:
        model: model to evaluate
        dataset: pytorch dataset
        batch_size: batch size
        loss_func: torch loss function
        device: device to evaluate on (cuda or cpu)
    Returns:
        loss: average loss over the dataset
    """
    #put model in eval mode and create dataloader
    model.eval()
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    total_loss = 0
    
    #loop over batches and compute the loss
    with torch.no_grad():
        for batch_X, batch_Y in tqdm(dataloader, desc = "Evaluating model", bar_format = "{l_bar}{bar:20}{r_bar}"):
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            prediction = model(batch_X)
            total_loss += loss_fn(prediction, batch_Y)
            
    return total_loss/len(dataloader)