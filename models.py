import torch
from abc import ABC, abstractmethod

class MyModel(torch.nn.Module, ABC):
    valid_features = ['waveforms', 'spectrograms', 'melspecs', 'mfcc']

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super(MyModel, self).__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, features: dict[str, torch.Tensor]):
        pass

    @property
    def device(self):
        return torch.device('cuda' if next(self.parameters()).is_cuda else 'cpu')


class SingleFeatureModel(MyModel, ABC):
    @abstractmethod
    def __init__(self, feature: str, *args, **kwargs):
        super(SingleFeatureModel, self).__init__(*args, **kwargs)
        if feature not in SingleFeatureModel.valid_features:
            raise ValueError(f'Feature name {feature} is not one of {SingleFeatureModel.valid_features}')
        self.feature = feature
    
    @abstractmethod
    def forward(self, features: dict[str, torch.Tensor]):
        pass

class MultiFeatureModel(MyModel, ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        super(MultiFeatureModel, self).__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, features: dict[str, torch.Tensor]):
        pass


    
class NilsHMeierCNN(SingleFeatureModel):
    """
    Implementation adapted from [Melspectrogram based CNN Classification by NilsHMeier on Kaggle](https://www.kaggle.com/code/nilshmeier/melspectrogram-based-cnn-classification)
    """
    def __init__(self, feature: str, out_features: int, *args, **kwargs):
        super(NilsHMeierCNN, self).__init__(feature, *args, **kwargs)

        self.out_features = out_features

        self.pipeline = torch.nn.Sequential(                
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same'),  
            torch.nn.ReLU(),    
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.3), 
            
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),  
            torch.nn.ReLU(),         
            torch.nn.MaxPool2d(kernel_size=2, stride=2),      

            torch.nn.Flatten(),                             
            torch.nn.Dropout1d(p=0.3),                  

            torch.nn.LazyLinear(out_features=64),   
            torch.nn.ReLU(),                           
            torch.nn.Linear(in_features=64, out_features=out_features),                         
        )   
        

    def forward(self, features: dict[str, torch.Tensor]):
        x = features[self.feature].unsqueeze(dim=1)
        x = self.pipeline(x)

        return x    # softmax will be applied in loss function
    

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional=False, *args, **kwargs):
        super(LSTMEncoder, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder = torch.nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional, 
        )

    def forward(self, x):
        outputs, hidden = self.encoder(x)

        return torch.cat(hidden, dim=0)
    
class Feedforward(torch.nn.Module):
    def __init__(self, num_features: int, hidden_size: int, *args, **kwargs):
        super(Feedforward, self).__init__(*args, **kwargs)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_features),
            torch.nn.ReLU(),
        )


    def forward(self, x):
        x = self.ff(x) + x

        return x
    
class LSTMClassifier(SingleFeatureModel):
    def __init__(self, feature: str, input_size: int, hidden_size: int, out_features: int, num_layers: int):
        super(LSTMClassifier, self).__init__(feature)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.num_layers = num_layers

        self.encoder = LSTMEncoder(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
        )
        self.classifier = torch.nn.Sequential(
            Feedforward(num_layers * hidden_size * 2, 64),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout1d(p=0.3),
            torch.nn.Linear(num_layers * hidden_size * 2, out_features),
        )

    def forward(self, features):
        x = features[self.feature]
        if self.feature in ('melspecs', 'spectrograms', 'mfcc'):
            x = x.permute(0, 2, 1)

        encoder_out = self.encoder(x)
        x = encoder_out.permute(1, 0, 2).flatten(1, 2)

        x = self.classifier(x)

        return x


class GRUEncoder(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, bidirectional=False, *args, **kwargs):
        super(GRUEncoder, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder = torch.nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional, 
        )

    def forward(self, x):
        outputs, hidden = self.encoder(x)

        return hidden
    
class GRUClassifier(SingleFeatureModel):
    def __init__(self, feature: str, input_size: int, hidden_size: int, out_features: int, num_layers: int):
        super(GRUClassifier, self).__init__(feature)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.num_layers = num_layers

        self.encoder = GRUEncoder(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
        )
        self.classifier = torch.nn.Sequential(
            Feedforward(num_layers * hidden_size, 64),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout1d(p=0.3),
            torch.nn.Linear(num_layers * hidden_size, out_features),
        )

    def forward(self, features):
        x = features[self.feature]
        if self.feature in ('melspecs', 'spectrograms', 'mfcc'):
            x = x.permute(0, 2, 1)

        encoder_out = self.encoder(x)
        x = encoder_out.permute(1, 0, 2).flatten(1, 2)

        x = self.classifier(x)

        return x
