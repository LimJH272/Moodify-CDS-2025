import torch
import os
from python_helpers import get_project_root_dir

feature_dir = os.path.join(get_project_root_dir(), 'processed', 'features')
label_dir = os.path.join(get_project_root_dir(), 'processed', 'label')

def save_processed_data(features: dict[str, torch.Tensor], labels: torch.Tensor, dset_name: str, train: bool):
    subdir = 'train' if train else 'eval'
    os.makedirs(os.path.join(feature_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(label_dir, subdir), exist_ok=True)

    fn = dset_name + '.pt'
    torch.save(features, os.path.join(feature_dir, subdir, fn))
    torch.save(labels, os.path.join(label_dir, subdir, fn))

    print(f'{dset_name} data saved successfully!')

def load_data(dset_name: str, train: bool):
    subdir = 'train' if train else 'eval'
    fn = dset_name + '.pt'
    features = torch.load(os.path.join(feature_dir, subdir, fn))
    labels = torch.load(os.path.join(label_dir, subdir, fn))

    return features, labels

def load_all_data(train: bool):
    subdir = 'train' if train else 'eval'
    feature_paths = sorted([os.path.join(feature_dir, subdir, f) for f in os.listdir(feature_dir) if os.path.splitext(f)[1] == '.pt'])
    label_paths = sorted([os.path.join(label_dir, subdir, f) for f in os.listdir(label_dir) if os.path.splitext(f)[1] == '.pt'])

    feature_ls = [torch.load(p) for p in feature_paths]
    feature_names = list(feature_ls[0].keys())
    features = {k: torch.cat([d[k] for d in feature_ls], dim=0) for k in feature_names}
    labels = torch.cat([torch.load(p) for p in label_paths], dim=0)

    return features, labels
