import torch
import torchvision.transforms.v2 as v2

from ml.dataset import VRNET2_Multi_Session_Dataset, ORIGNAL_COLS_TO_PREDICT
from ml.models import EfficientnetEBMAgent, NCELoss
from ml.training import EBMTrainer


#define training and validation datasets
im_size = (456, 456)

x_transform = v2.Compose([
        v2.Resize(im_size),
    ])


def y_transform(y):
    norm_factor = torch.tensor([3, 3, 3, 10, 10, 10, 1, 1, 1, 1])
    y = y / norm_factor
    return y

train_dataset = VRNET2_Multi_Session_Dataset("train", 
                                            seq_length=1, 
                                            cols_to_predict=ORIGNAL_COLS_TO_PREDICT,
                                            transform=x_transform,
                                            target_transform=y_transform)

val_dataset = VRNET2_Multi_Session_Dataset("val", 
                                            seq_length=1, 
                                            cols_to_predict=ORIGNAL_COLS_TO_PREDICT,
                                            transform=x_transform,
                                            target_transform=y_transform)

#define model
model = EfficientnetEBMAgent(im_size=im_size, control_input_size=10, feature_vector_size=512, size=4)

#hyperparmaters
lr = 1e-5
loss_func = NCELoss()
weight_decay = 1e-4
schedular = (torch.optim.lr_scheduler.ExponentialLR, {"gamma":0.95})

#select device
gpu_num = 6
device = torch.device(f'cuda:{gpu_num}')
print(f"Using device {gpu_num}: {torch.cuda.get_device_properties(gpu_num).name}")

#setup training 
trainer = EBMTrainer(name = "ENB4_EBM_TEST",
                  model = model,
                  train_dataset = train_dataset,
                  val_dataset = val_dataset,
                  epochs = 20,
                  batch_size = 64,
                  lr = lr,
                  loss_fn = loss_func,
                  weight_decay = weight_decay,
                  save_freq = 5,
                  device = device,
                  workers = 32,
                  scheduler=schedular
                  )

#run trainer
trainer.train()