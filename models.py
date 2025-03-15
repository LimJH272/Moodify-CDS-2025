import torch
    
class KaggleCNN(torch.nn.Module):
    """
    Implementation adapted from [Melspectrogram based CNN Classification by NilsHMeier on Kaggle](https://www.kaggle.com/code/nilshmeier/melspectrogram-based-cnn-classification)
    """
    def __init__(self):
        super().__init__()

        self.pipeline = torch.nn.Sequential(                # input shape = (1, 256, 321)
            torch.nn.Conv2d(1, 16, 3, padding='same'),      # output shape = (16, 256, 321)
            torch.nn.MaxPool2d(2, 2),                       # output shape = (16, 128, 160)
            torch.nn.Conv2d(16, 32, 3, padding='same'),     # output shape = (32, 128, 160)
            torch.nn.MaxPool2d(2, 2),                       # output shape = (32, 64, 80)
            torch.nn.Flatten(),                             # output shape = (163840,)
            torch.nn.Dropout1d(0.3),                        # output shape = (163840,)
            torch.nn.Linear(163840, 64),                    # output shape = (64,)
            torch.nn.Linear(64, 4),                         # output shape = (4,)
        )                                                                    
        

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)

        x = self.pipeline(x)

        return x    # softmax will be applied in loss function
    