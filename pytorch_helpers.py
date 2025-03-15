import torch
import os
from python_helpers import get_project_root_dir

melspec_dir = os.path.join(get_project_root_dir(), 'processed', 'melspec')
label_dir = os.path.join(get_project_root_dir(), 'processed', 'label')

def save_processed_data(melspecs: torch.Tensor, labels: torch.Tensor, dset_name: str):
    os.makedirs(melspec_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    fn = dset_name + '.pt'
    torch.save(melspecs, os.path.join(melspec_dir, fn))
    torch.save(labels, os.path.join(label_dir, fn))

    print(f'{dset_name} data saved successfully!')

def load_data(dset_name: str):
    fn = dset_name + '.pt'
    melspecs = torch.load(os.path.join(melspec_dir, fn))
    labels = torch.load(os.path.join(label_dir, fn))

    return melspecs, labels

def load_all_data():
    melspec_paths = sorted([os.path.join(melspec_dir, f) for f in os.listdir(melspec_dir) if os.path.splitext(f)[1] == '.pt'])
    label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if os.path.splitext(f)[1] == '.pt'])

    melspecs = torch.cat([torch.load(p) for p in melspec_paths], dim=0)
    labels = torch.cat([torch.load(p) for p in label_paths], dim=0)

    return melspecs, labels
