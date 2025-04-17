import torch, librosa
import matplotlib.pyplot as plt

def plot_waveform(waveform, sr, title=None, ax=None):
    waveform = waveform.numpy()

    num_frames = waveform.shape[-1]
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(time_axis, waveform, linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    if title is not None:
        ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None, to_db=False):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram) if to_db else specgram, origin="lower", aspect="auto", interpolation="nearest")