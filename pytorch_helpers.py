import torch, torchaudio
import os
from python_helpers import get_project_root_dir

def save_processed_data(melspecs: torch.Tensor, labels: torch.Tensor, dset_name: str):
    melspec_dir = os.path.join(get_project_root_dir(), 'processed', 'melspec')
    label_dir = os.path.join(get_project_root_dir(), 'processed', 'label')

    os.makedirs(melspec_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    torch.save(melspecs, os.path.join(melspec_dir, dset_name + '.pt'))
    torch.save(labels, os.path.join(label_dir, dset_name + '.pt'))
    