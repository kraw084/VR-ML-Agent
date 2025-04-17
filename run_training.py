import torch
import torchvision.transforms.v2 as v2

from ml.dataset import VRNET2_Multi_Session_Dataset, ORIGNAL_COLS_TO_PREDICT
from ml.models import EfficientnetAgent
from ml.training import Trainer


#define training and validation datasets
im_size = (288, 288)

x_transform = v2.Compose([
        v2.Resize(im_size),
    ])

train_dataset = VRNET2_Multi_Session_Dataset("train", 
                                            seq_length=1, 
                                            cols_to_predict=ORIGNAL_COLS_TO_PREDICT,
                                            transform=x_transform)

val_dataset = VRNET2_Multi_Session_Dataset("val", 
                                            seq_length=1, 
                                            cols_to_predict=ORIGNAL_COLS_TO_PREDICT,
                                            transform=x_transform)

#define model
model = EfficientnetAgent(im_size=im_size, output_size=10)

#hyperparmaters
lr = 1e-4
loss_func = torch.nn.MSELoss()
weight_decay = 1e-3

#select device
gpu_num = 0
device = torch.device(f'cuda:{gpu_num}')
print(f"Using device {gpu_num}: {torch.cuda.get_device_properties(gpu_num).name}")

#setup training 
trainer = Trainer(name = "ENB2_MSE",
                  model = model,
                  train_dataset = train_dataset,
                  val_dataset = val_dataset,
                  epochs = 100,
                  batch_size = 128,
                  lr = lr,
                  loss_fn = loss_func,
                  weight_decay = weight_decay,
                  save_freq = 10,
                  device = device,
                  workers = 32
                  )

#run trainer
trainer.train()