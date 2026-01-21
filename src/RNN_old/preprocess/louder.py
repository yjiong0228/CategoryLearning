from pydub import AudioSegment
import glob
from tqdm import tqdm
wav_files = glob.glob('Recording/26/*/*.wav')
for wav_file in tqdm(wav_files):
    audio = AudioSegment.from_wav(wav_file)
    audio = audio + 6  # Increase volume by 10 dB
    audio.export(wav_file, format="wav")