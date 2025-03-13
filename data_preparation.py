import numpy as np
import os
import pandas as pd
import json

import torch, torchaudio

from helpers import *

data_subdirs = {
    'SoundTracks': ['Set1', 'Set2'],
}

TARGET_SAMPLE_RATE = 44100
TARGET_AUDIO_LENGTH = 15 * TARGET_SAMPLE_RATE

def get_main_data_dir():
    return os.path.join(get_project_root_dir(), 'data')

def get_data_subdir_paths(data_subdir_map: dict[str, (dict|list)], curr_dir: str=get_main_data_dir()) -> list[str]:
    data_subdir_paths = []
    for dset, subdirs in data_subdir_map.items():
        if isinstance(subdirs, dict):
            data_subdir_paths.extend(get_data_subdir_paths(subdirs, os.path.join(curr_dir, dset)))
        elif isinstance(subdirs, list):
            data_subdir_paths.extend([os.path.join(curr_dir, dset, d) for d in subdirs])
        else:
            raise ValueError('Why are you here?', dset, subdirs)
    
    return data_subdir_paths

def get_audio_files_paths(data_dir_paths_ls: list[str]) -> list[str]:
    allowed_extensions = ['.mp3', '.wav']
    return [os.path.join(dir_path, file) for dir_path in data_dir_paths_ls for file in os.listdir(dir_path) if os.path.splitext(file)[1] in allowed_extensions]

def split_waveform_segments(wf: torch.Tensor, target_len: int, k: int=TARGET_SAMPLE_RATE) -> list[torch.Tensor]:
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

class Stereo2Mono(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.tensor([0.5, 0.5], requires_grad=False).view(2, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.sum(waveform * self.weights, dim=0) * np.sqrt(2)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    audio_tensors = []
    for path in get_audio_files_paths(get_data_subdir_paths(data_subdirs)):
        print(path + '...', end='\t')

        waveform, sr = torchaudio.load(path)
        if sr != TARGET_SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, TARGET_SAMPLE_RATE)(waveform)
        mono_wf = Stereo2Mono()(waveform)
        wf_segments = split_waveform_segments(mono_wf, TARGET_AUDIO_LENGTH)
        audio_tensors.extend(wf_segments)

        print('Done!')
    
    padded_audio_tensors = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True).to(device)
    print(padded_audio_tensors.shape)

    melspecs = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_mels=512,
        n_fft=8192, 
        center=False,
    ).to(device)(padded_audio_tensors)
    print(melspecs.shape)
    
    torch.save(melspecs.cpu(), 'soundtracks_melspecs.pt')
    print('Saved processed melspecs')
