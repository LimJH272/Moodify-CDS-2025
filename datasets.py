import torch
import pytorch_helpers as pth

class AllDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        melspecs, labels = pth.load_all_data()
        assert melspecs.shape[0] == labels.shape[0]

        self.melspecs = melspecs
        self.labels = labels

    def __len__(self):
        return self.melspecs.shape[0]
    
    def __getitem__(self, idx: int):
        return self.melspecs[idx, ...], self.labels[idx]
    
class SoundTracksDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        
        melspecs, labels = pth.load_data("SoundTracks")
        assert melspecs.shape[0] == labels.shape[0]

        self.melspecs = melspecs
        self.labels = labels

    def __len__(self):
        return self.melspecs.shape[0]
    
    def __getitem__(self, idx: int):
        return self.melspecs[idx, ...], self.labels[idx]