import torch
from abc import ABC, abstractmethod

class MyModel(torch.nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super(MyModel, self).__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor):
        return x

    @property
    def device(self):
        return torch.device('cuda') if next(self.parameters()).is_cuda else torch.device('cpu')
    
class NilsHMeierCNN(MyModel):
    """
    Implementation adapted from [Melspectrogram based CNN Classification by NilsHMeier on Kaggle](https://www.kaggle.com/code/nilshmeier/melspectrogram-based-cnn-classification)
    """
    def __init__(self):
        super(NilsHMeierCNN, self).__init__()

        self.pipeline = torch.nn.Sequential(                
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same'),  
            torch.nn.ReLU(),    
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # torch.nn.Dropout2d(p=0.3), 
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),  
            torch.nn.ReLU(),         
            torch.nn.MaxPool2d(kernel_size=2, stride=2),      

            torch.nn.Flatten(),                             
            torch.nn.Dropout1d(p=0.3),                  

            torch.nn.Linear(in_features=163840, out_features=64),   
            torch.nn.ReLU(),                           
            torch.nn.Linear(in_features=64, out_features=4),                         
        )   
        

    def forward(self, x: torch.Tensor):
        x = self.pipeline(x)

        return x    # softmax will be applied in loss function
    