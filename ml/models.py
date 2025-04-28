import torch
import torchvision

#from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from efficientnet_pytorch import EfficientNet

class EfficientnetAgent(torch.nn.Module):
    def __init__(self, im_size = (288, 288), output_size = 10, pretrained = True, size=2):
        #self.model = efficientnet_b2(num_classes = 1000, weights = None if not pretrained else EfficientNet_B2_Weights.IMAGENET1K_V1)
        super().__init__()
        
        if pretrained:
            self.model = EfficientNet.from_pretrained(f"efficientnet-b{size}") 
        else:
            self.model = EfficientNet.from_name(f"efficientnet-b{size}") 
            
        self.model._fc = torch.nn.Linear(self.model._fc.in_features, output_size)
        
        self.size = size
        self.im_size = im_size
        self.output_size = output_size
        
    def forward(self, input):
        return self.model.forward(input)   
    
    def str(self):
        return f"EfficentNet B{self.size} - Im Size: {self.im_size} - Output Size: {self.output_size}"     

