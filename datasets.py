import torch
import pytorch_helpers as pth
from abc import abstractmethod, ABC
import sklearn.model_selection as skl

class MelSpecDataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def __init__(self, melspecs: torch.Tensor, labels: torch.Tensor):
        super(MelSpecDataset, self).__init__()

        # Assumption: melspecs has shape (N,H,W) or (N,C,H,W)
        assert melspecs.shape[0] == labels.shape[0] and 3 <= len(melspecs.shape) <= 4
        self.melspecs = melspecs
        self.labels = labels

    def __len__(self):
        return self.melspecs.shape[0]
    
    def __getitem__(self, idx: int):
        return self.melspecs[idx, ...], self.labels[idx]

    def NCHW_to_NHW(self):
        if len(self.melspecs.shape) == 4 and self.melspecs.shape[1] == 1:
            self.melspecs = self.melspecs.squeeze(dim=1)
        else:
            print('Conversion to NHW failed: melspecs is not a 4D tensor OR size of dim 1 is not 1')
        
        return self

    def NHW_to_NCHW(self):
        if len(self.melspecs.shape) == 3:
            self.melspecs = self.melspecs.unsqueeze(dim=1)
        else:
            print('Conversion to NCHW failed: melspecs is not a 3D tensor')
        
        return self
    
    def train_test_split(self, split_size: int|float=0.2):
        train_melspecs, test_melspecs, train_labels, test_labels = skl.train_test_split(self.melspecs, self.labels, test_size=split_size, shuffle=True)

        return MelSpecDataset(train_melspecs, train_labels), MelSpecDataset(test_melspecs, test_labels)


class AllDataset(MelSpecDataset):
    def __init__(self):
        melspecs, labels = pth.load_all_data()
        super(AllDataset, self).__init__(melspecs, labels)
    
class SoundTracksDataset(MelSpecDataset):
    def __init__(self):
        melspecs, labels = pth.load_data("SoundTracks")
        super(SoundTracksDataset, self).__init__(melspecs, labels)