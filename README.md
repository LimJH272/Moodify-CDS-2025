# Moodify: Emotion prediction from music - with playlist guidance (VGG)
SUTD 50.038 Computational Data Science Project (2025)
The branch is for the VGG model

### SoundTracks
1. Download the dataset [here](https://osf.io/p6vkg/)
2. Extract the ZIP file
3. Rename the folder as `SoundTracks`
4. Extract the ZIP files named `Set1`, `set2` and `1min`
5. Move/copy the folder to the `data` directory

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

### 4. Run the data_preparation.py file
```cmd
python data_preparation.py
```

### 5. Run throught the main-vgg.ipynb file

