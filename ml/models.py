import torch
import torchvision
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights


class EfficientnetAgent(efficientnet_b2):
    def __init__(self, im_size = (288, 288), output_size = 10, pretrained = True):
        super().__init__(num_classes = output_size, weights = None if not pretrained else EfficientNet_B2_Weights.IMAGENET1K_V1)
        
        self.im_size = im_size
        self.output_size = output_size
        
