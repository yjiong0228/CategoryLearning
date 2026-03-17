import glob
import torchaudio
import numpy as np
from tqdm import tqdm

import whisper
import pandas as pd

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# 加载Whisper预训练模型
model = whisper.load_model("turbo")
print("Model loaded!")



recording_dirs = glob.glob('Recording/26/*')
# recording_dirs = [x for x in recording_dirs if x.split('/')[-1].split('_')[0] in ['8', '13', '15', '19', '20', '21']]
# recording_dirs.sort(key=lambda x: (int(x.split('/')[-1].split('_')[0]), int(x.split('/')[-1].split('_')[1].split('.')[0])))
print(recording_dirs)
for recording_dir in tqdm(recording_dirs, desc='recording processing', leave=True):
    iSub = recording_dir.split('/')[-1].split('_')[0]
    iSession = recording_dir.split('/')[-1].split('_')[1]
    audio_files = glob.glob(f'{recording_dir}/*.wav')
    audio_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    # audio_files = [x for x in audio_files if 112 <= int(x.split('/')[-1].split('.')[0]) <= 169]
    df = pd.DataFrame(columns=['iSub', 'iSession', 'iTrial', 'text'])
    for audio_file in tqdm(audio_files, desc='sub processing', leave=False):
        iTrial = audio_file.split('/')[-1].split('.')[0]
        waveform, sample_rate = torchaudio.load(audio_file)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        audio = waveform.squeeze().numpy()
        result = model.transcribe(audio)
        df = pd.concat([df, pd.DataFrame([{'iSub': iSub, 'iSession': iSession, 'iTrial': iTrial, 'text': result['text']}])], ignore_index=True)
    if os.path.exists('recording-424.csv'):
        df.to_csv('recording-424.csv', index=False, mode='a', header=False)
    else:        
        df.to_csv('recording-424.csv', index=False, mode='w', header=True)
        
    