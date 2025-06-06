import torch
from abc import ABC, abstractmethod
import torch.nn as nn


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
            torch.nn.Conv2d(in_channels=1, out_channels=16,
                            kernel_size=3, padding='same'),
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
            torch.nn.Linear(in_features=64, out_features=4),                         
        )   
        

    def forward(self, features: dict[str, torch.Tensor]):
        x = features[self.feature].unsqueeze(dim=1)
        x = self.pipeline(x)

        return x    # softmax will be applied in loss function


class VGGStyleCNN(SingleFeatureModel):
    """VGG-inspired CNN with batch normalization for 128x323 melspectrograms"""

    def __init__(self, feature: str):
        super().__init__(feature)

        self.cnn = torch.nn.Sequential(
            # Block 1
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            # Block 2
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            # Block 3
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256 * 16 * 40, 512),  # Adjusted for 128x323 input
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 4)
        )

    def forward(self, features: dict[str, torch.Tensor]):
        x = features[self.feature].unsqueeze(1)  # Add channel dimension
        x = self.cnn(x)
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ImprovedEmotionTransformer(SingleFeatureModel):
    def __init__(self, input_dim=128, num_classes=4, d_model=128, nhead=8, num_layers=4, dropout=0.7):
        super().__init__(feature='melspecs')
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=512, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, features):
        x = features[self.feature]  # Get melspecs from input dict
        x = x.permute(0, 2, 1)  # (B, T, F)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # (T, B, D)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        x = self.dropout(x)
        return self.fc_out(x)

    

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
            dropout=(0.5 if num_layers > 1 else 0.0)
        )

    def forward(self, x):
        outputs, hidden = self.encoder(x)

        return torch.cat(hidden, dim=0)
    
class Feedforward(torch.nn.Module):
    def __init__(self, num_features: int, hidden_size: int, *args, **kwargs):
        super(Feedforward, self).__init__(*args, **kwargs)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_size),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Linear(hidden_size, num_features),
            torch.nn.LeakyReLU(negative_slope=0.01),
        )


    def forward(self, x):
        x = self.ff(x) + x

        return x
    
class LSTMClassifier(SingleFeatureModel):
    def __init__(self, feature: str, input_size: int, hidden_size: int, out_features: int, num_layers: int, bidirectional=False):
        super(LSTMClassifier, self).__init__(feature)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder = LSTMEncoder(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            bidirectional=bidirectional,
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear((2 if bidirectional else 1) * num_layers * hidden_size * 2, 64),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout1d(p=0.5),
            torch.nn.Linear(64, out_features),
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
            dropout=(0.5 if num_layers > 1 else 0.0)
        )

    def forward(self, x):
        outputs, hidden = self.encoder(x)

        return hidden
    
class GRUClassifier(SingleFeatureModel):
    def __init__(self, feature: str, input_size: int, hidden_size: int, out_features: int, num_layers: int, bidirectional=False):
        super(GRUClassifier, self).__init__(feature)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.encoder = GRUEncoder(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            bidirectional=bidirectional
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear((2 if bidirectional else 1) * num_layers * hidden_size, 64),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout1d(p=0.5),
            torch.nn.Linear(64, out_features),
        )

    def forward(self, features):
        x = features[self.feature]
        if self.feature in ('melspecs', 'spectrograms', 'mfcc'):
            x = x.permute(0, 2, 1)

        encoder_out = self.encoder(x)
        x = encoder_out.permute(1, 0, 2).flatten(1, 2)

        x = self.classifier(x)

        return x

class CNN_RNN(SingleFeatureModel):
    def __init__(self, feature: str, out_features: int, gru_args: dict, *args, **kwargs):
        super().__init__(feature, *args, **kwargs)

        self.feature = feature

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding='same'),  
            torch.nn.LeakyReLU(negative_slope=0.01),    
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.3),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),  
            torch.nn.LeakyReLU(negative_slope=0.01),  
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout2d(p=0.3),

            torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding='same'),  
            torch.nn.LeakyReLU(negative_slope=0.01), 
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            # torch.nn.Flatten(),
        )
        self.rnn = GRUEncoder(**gru_args)
        self.classifier =  torch.nn.Sequential(
            torch.nn.Linear((2 if gru_args['bidirectional'] else 1) * gru_args['num_layers'] * gru_args['hidden_size'], 64),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.Dropout1d(p=0.3),
            torch.nn.Linear(64, out_features),
        )

    def forward(self, features):
        x = features[self.feature].unsqueeze(dim=1)
        x = self.cnn(x)

        x = x.squeeze(1).permute(0, 2, 1)
        x = self.rnn(x)

        x = x.permute(1, 0, 2).flatten(1, 2)
        x = self.classifier(x)

        return x