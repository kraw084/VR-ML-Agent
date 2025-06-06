import torch
import torchvision

from efficientnet_pytorch import EfficientNet

class EfficientnetAgent(torch.nn.Module):
    def __init__(self, im_size = (288, 288), output_size = 10, pretrained = True, size=2):
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
    
    def __str__(self):
        return f"EfficentNet B{self.size} - Im Size: {self.im_size} - Output Size: {self.output_size}"     
    
      
    
class ControlVecFeatureExtractor(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size)
        )
        
    def forward(self, input):
        return self.layers(input)


class EMBHead(torch.nn.Module):
    def __init__(self, feature_vector_size):
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2 * feature_vector_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
        
    def forward(self, input):
        x = torch.concat(input, dim=-1)
        return self.layers(x)
     
    
class EfficientnetEBMAgent(torch.nn.Module):
    def __init__(self, im_size = (288, 288), control_input_size = 10, feature_vector_size = 512, pretrained = True, size=2):
        super().__init__()
        
        self.size = size
        self.im_size = im_size
        self.control_input_size = control_input_size
        self.feature_vector_size = feature_vector_size
        
        if pretrained:
            self.im_feature_extractor = EfficientNet.from_pretrained(f"efficientnet-b{size}") 
        else:
            self.im_feature_extractor = EfficientNet.from_name(f"efficientnet-b{size}") 
        self.im_feature_projection = torch.nn.Linear(self.im_feature_extractor._fc.in_features, feature_vector_size)
            
        self.control_feature_extractor = ControlVecFeatureExtractor(control_input_size, feature_vector_size)

        self.head = EMBHead(feature_vector_size)
        
    def extract_image_feature(self, input):
        x = self.im_feature_extractor.extract_features(input)
        x = self.im_feature_extractor._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.im_feature_extractor._dropout(x)
        x = self.im_feature_projection(x)

        return x
    
    def extract_control_feature(self, input):
        return self.control_feature_extractor(input)
    
    def run_head(self, im_feature, control_feature):
        return self.head((im_feature, control_feature))
       
        
    def forward(self, input):
        image, control_vector = input
        im_feature = self.extract_image_feature(image) 
        control_feature = self.extract_control_feature(control_vector)
        return self.run_head(im_feature, control_feature)
    
    def __str__(self):
        return f"EfficentNet B{self.size} EBM - Im Size: {self.im_size} - Control Size: {self.control_input_size} - Feature Vector Size: {self.feature_vector_size}"
    
    
class NCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        
    def forward(self, energy, fake_energy):
        #energy [b, 1]
        #fake: [b, n, 1]
        
        all_energies = torch.cat([energy, fake_energy.squeeze(-1)], dim=1)
        labels = torch.zeros(all_energies.shape[0], dtype=torch.long, device=all_energies.device)
        
        return self.ce_loss(-all_energies, labels)
  
"""  
model = EfficientnetEBMAgent()

B = 16
con_in = torch.rand((B, 10))
im_in = torch.rand((B, 3, 288, 288))

en = model((im_in, con_in))

negs = torch.rand(B, 10, 1)

loss_func = NCELoss()
loss = loss_func(en, negs)
loss.backward()

for name, param in model.named_parameters():
    if param.grad is None:
        print(f"{name}: ❌ No gradient (None)")
    elif torch.isnan(param.grad).any():
        print(f"{name}: ❌ Gradient has NaNs")
    elif torch.all(param.grad == 0):
        print(f"{name}: ⚠️ Gradient is all zero")
    else:
        print(f"{name}: ✅ grad norm = {param.grad.norm():.4f}")
"""