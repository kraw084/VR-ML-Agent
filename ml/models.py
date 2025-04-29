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
        
    def foward(self, input):
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
        
    def foward(self, input):
        x = torch.concat(input, dim=1)
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
        
    def forward(self, input):
        image, control_vector = input
        im_feature = self.extract_image_feature(image) 
        control_feature = self.control_feature_extractor(control_vector)
        energy = self.head((im_feature, control_feature))
        
        return energy
    
    def __str__(self):
        return f"EfficentNet B{self.size} - Im Size: {self.im_size} - Output Size: {self.output_size}"