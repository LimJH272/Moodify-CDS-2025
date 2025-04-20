# Moodify: Emotion prediction from music - with playlist guidance
SUTD 50.038 Computational Data Science Project (2025)

## NOTE
This is the main branch, which contains the baseline CNN model, RNN models, and CNN+RNN model.

For other models, please look at the following branches:
- [Transformer-Yash](https://github.com/LimJH272/Moodify-CDS-2025/tree/Transformer-Yash)
- [VGG-Jignesh](https://github.com/LimJH272/Moodify-CDS-2025/tree/VGG-Jignesh)

## Table of Contents
- [Required Datasets](#required-datasets)
- [Installation & Setup](#installation--setup)

## Required Datasets

### SoundTracks
1. Download the dataset [here](https://osf.io/p6vkg/)
2. Extract the ZIP file
3. Rename the folder as `SoundTracks`
4. Extract the ZIP files named `Set1`, `set2` and `1min`
5. Move/copy the folder to the `data` directory

### Emotify
1. Download the dataset [here](http://www2.projects.science.uu.nl/memotion/emotifydata/)
2. Extract the ZIP file
3. Save the extracted contents in a folder named `Emotify`
4. Move/copy the folder to the `data` directory

## Installation & Setup
- Python (version 3.12 or lower)
- PyTorch (torch, torchaudio, torchvision)

### 1. Clone this repository onto your local machine
```cmd
git clone https://github.com/LimJH272/Moodify-CDS-2025.git
```

### 2. Create virtual environment (conda recommended)
```cmd
conda create -n python312 python=3.12 anaconda
conda activate python312
```

### 3. Install Python dependencies
```cmd
pip install -r requirements.txt
```

### 4. Install PyTorch from the [official page](https://pytorch.org/get-started/locally/)

### 5. Prepare datasets
```cmd
python data_preparation.py
```

## Running the Notebooks
The following notebooks may be run after the data preparation step:
- `data_visualisation.ipynb`
- `jignesh_vgg.ipynb`
- `yash_1.ipynb` (Should be run on Google Colab, remove Colab-proprietary cells if running elsewhere)
- `RNN.ipynb`
- `CNN_RNN.ipynb`