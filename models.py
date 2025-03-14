import torch

class BN_Conv2d_ReLU(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int], stride=1, padding='valid', dropout=0.0):
        super().__init__()

        self.batchnorm = torch.nn.BatchNorm2d(in_channels)
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True, padding_mode='replicate')
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor):
        x = self.batchnorm(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x
    
class BN_Linear_ReLU(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout=0.0):
        super().__init__()

        self.batchnorm = torch.nn.BatchNorm1d(in_features)
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor):
        x = self.batchnorm(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class MelSpecCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

                                                                            # input shape = (1, 512, 160)
        self.conv1 = BN_Conv2d_ReLU(1, 4, (5,3), dropout=0.1)               # output shape = (4, 508, 158)
        self.conv2 = BN_Conv2d_ReLU(4, 16, (5,3), dropout=0.1)              # output shape = (16, 504, 156)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(3,2), stride=(3,2)) # output shape = (16, 168, 78)

        self.conv3 = BN_Conv2d_ReLU(16, 8, (5,3), dropout=0.1)              # output shape = (8, 164, 76)
        self.conv4 = BN_Conv2d_ReLU(8, 4, (5,3), dropout=0.1)               # output shape = (4, 160, 74)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)         # output shape = (4, 80, 37)
        self.flatten = torch.nn.Flatten()                                   # output shape = (11840,)

        self.fc_1 = BN_Linear_ReLU(11840, 1000, dropout=0.1)                # output shape = (1000,)
        self.fc_2 = BN_Linear_ReLU(1000, 100, dropout=0.1)                  # output shape = (100,)
        self.fc_3 = BN_Linear_ReLU(100, 4, dropout=0.1)                     # output shape = (4,)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)      # input is a 3D tensor of shape (batch_size, height, width), but Conv2d expects shape (batch_size, num_channels, height, width)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool2(x)
        x = self.flatten(x)

        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)

        return x    # softmax will be applied in loss function