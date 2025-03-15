import torch
    
class NilsHMeierCNN(torch.nn.Module):
    """
    Implementation adapted from [Melspectrogram based CNN Classification by NilsHMeier on Kaggle](https://www.kaggle.com/code/nilshmeier/melspectrogram-based-cnn-classification)
    """
    def __init__(self):
        super().__init__()

        self.pipeline = torch.nn.Sequential(                
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same'),      
            torch.nn.MaxPool2d(kernel_size=2, stride=2),  

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),     
            torch.nn.MaxPool2d(kernel_size=2, stride=2),      

            torch.nn.Flatten(),                             
            torch.nn.Dropout1d(p=0.3),                  
                  
            torch.nn.LazyLinear(out_features=64),                        
            torch.nn.Linear(in_features=64, out_features=4),                         
        )
        

    def forward(self, x: torch.Tensor):
        x = self.pipeline(x)

        return x    # softmax will be applied in loss function
    