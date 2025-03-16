import numpy as np
import os
import pandas as pd
import json

import torch, torchaudio
import librosa

import python_helpers as pyh
import pytorch_helpers as pth

ROOT_DIR = pyh.get_project_root_dir()
TARGET_SAMPLE_RATE = 44100
TARGET_AUDIO_LENGTH = 15 * TARGET_SAMPLE_RATE

LABEL_TO_INT = {
    "Happy": 0,
    "Sad": 1,
    "Anger": 2,
    "Neutral": 3,
}
INT_TO_LABEL = {v: k for k,v in LABEL_TO_INT.items()}

def get_labels_from_file(path: str, audio_ref_col: str, label_col: str, label_cats: list[str], label_mapping: dict[str, str]) -> pd.DataFrame:
    _, file_ext = os.path.splitext(path)
    if file_ext == '.csv':
        df = pd.read_csv(path, low_memory=False, dtype={audio_ref_col: str})
    elif file_ext == '.xls' or file_ext == '.xlsx':
        df = pd.read_excel(path, dtype={audio_ref_col: str})

    df = df.loc[df[label_col].apply(lambda l: l in label_cats), [audio_ref_col, label_col]]
    df[label_col] = df[label_col].apply(lambda l: label_mapping[l])
    
    return df

class Stereo2Mono(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.tensor([0.5, 0.5], requires_grad=False).view(2, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if len(waveform.shape) == 2 and waveform.shape[0] == 2:
            return torch.sum(waveform * self.weights, dim=0) * np.sqrt(2)
        else:
            raise ValueError(f'{waveform.shape} is not a stereo waveform')
        
def path_to_waveform_tensor(path: str, sample_rate=TARGET_SAMPLE_RATE):
    waveform, sr = torchaudio.load(path)

    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    
    mono_wf = Stereo2Mono()(waveform)

    return mono_wf.cpu()

def get_wf_label_from_config(config: list[dict], root=ROOT_DIR, label_mapping=LABEL_TO_INT):
    wf_ls, label_ls = [], []

    for dset in config:
        audios_path = os.path.join(root, dset["audios_rel_path"])
        labels_path = os.path.join(root, dset["labels_rel_path"])
        labels_df = get_labels_from_file(
            path=labels_path,
            audio_ref_col=dset['audio_ref_col'],
            label_col=dset['label_col'],
            label_cats=dset['label_categories'],
            label_mapping=dset['label_mapping'],
        )

        audio_filenames = [str(name) + dset['audio_format'] for name in labels_df[dset['audio_ref_col']]]
        audio_path_ls = [os.path.join(audios_path, fn) for fn in audio_filenames]
        audio_wf_ls = [path_to_waveform_tensor(path) for path in audio_path_ls]

        wf_ls.extend(audio_wf_ls)
        label_ls.extend([label_mapping[l] for l in labels_df[dset['label_col']]])
    
    return wf_ls, label_ls

def split_waveform_segments(wf: torch.Tensor, target_len=TARGET_AUDIO_LENGTH, k: int=TARGET_AUDIO_LENGTH // 3) -> list[torch.Tensor]:
    assert len(wf.shape) == 1

    wf_len = wf.shape[0]

    if wf_len < target_len:
        return [wf.detach().clone()]
    else:
        tensor_ls = []
        for i in range(int(np.ceil(round(wf_len / target_len, 7)))):
            if wf_len - i * target_len > target_len:
                tensor_ls.append(wf[i*target_len:(i+1)*target_len])
            else:
                if wf_len - i * target_len > k:
                    tensor_ls.append(wf[-target_len:])
    
    return tensor_ls

def segmentate_waveforms(wf_ls: list[torch.Tensor], label_ls: list[int]):
    new_wf_ls, new_label_ls = [], []

    for wf, label in zip(wf_ls, label_ls):
        wf_segments = split_waveform_segments(wf)

        new_wf_ls.extend(wf_segments)
        new_label_ls.extend([label] * len(wf_segments))
    
    return new_wf_ls, new_label_ls

class NormaliseMelSpec(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.normalise = lambda x: torch.from_numpy(librosa.power_to_db(x, ref=np.max))

    def forward(self, x):
        return self.normalise(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_config: list[dict] = json.load(open('data_config.json', 'r'))

    for dset_config in data_config:
        dset_name = dset_config['dataset_name']

        wf_ls, label_ls = get_wf_label_from_config(dset_config['subsets'])
        wf_ls, label_ls = segmentate_waveforms(wf_ls, label_ls)
        
        wf_tensor = torch.nn.utils.rnn.pad_sequence(wf_ls, batch_first=True)
        label_tensor = torch.tensor(label_ls)

        n_mels = 256
        melspec_tensor = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SAMPLE_RATE,
            n_mels=n_mels,
            n_fft=n_mels * 16, 
            center=True,
        ).to(device)(wf_tensor.to(device)).cpu()
        melspec_tensor = NormaliseMelSpec()(melspec_tensor)

        print(dset_name)
        print(melspec_tensor.shape, label_tensor.shape)
        print(torch.bincount(label_tensor))

        pth.save_processed_data(melspec_tensor, label_tensor, dset_name)
