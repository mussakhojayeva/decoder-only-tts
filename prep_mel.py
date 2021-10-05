import glob, os, soundfile, re, librosa, wave, contextlib
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import params as hp
from pathlib import Path
import pandas as pd

def get_duration(file_path):
    duration = None
    if os.path.exists(file_path) and Path(file_path).stat().st_size > 0:
        with contextlib.closing(wave.open(file_path,'r')) as f:
            frames = f.getnframes()
            if frames>0:
                rate = f.getframerate()
                duration = frames / float(rate)
    return duration if duration else 0


def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=None,
    fmax=None,
    eps=1e-10,
    log_base=10.0,
):

    x_stft = librosa.stft(
        audio,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=win_length,
        window=window,
        pad_mode="reflect",
    )
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)
    mel = np.maximum(eps, np.dot(spc, mel_basis.T))

    if log_base is None:
        return np.log(mel)
    elif log_base == 10.0:
        return np.log10(mel)
    elif log_base == 2.0:
        return np.log2(mel)
    else:
        raise ValueError(f"{log_base} is not supported.")

        

def main():
    
    wav_dir = 'data/RUSLAN_24k'
    dumpdir = "data/RUSLAN_dump"
    df = pd.read_csv('data/metadata_RUSLAN_22200.csv', sep='|', header=None, names=['id', 'value'])
    
    def process_text(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    df.value = df.value.apply(process_text)
    utt2text = dict(zip(df.id, df.value))
    print(df)
    return
    wav_files =  glob.glob(wav_dir + '/*.wav')

    scaler = StandardScaler()
    max_dur = 20
    utt2mel = {}
    for wav in tqdm(wav_files):
        if get_duration(wav) > max_dur: continue
        audio, sample_rate = soundfile.read(wav)
        utt_id = os.path.basename(wav).replace('.wav', '')
        mel = logmelfilterbank(
            audio,
            sampling_rate=sample_rate,
            hop_size=hp.hop_size, #300
            fft_size=hp.fft_size, #2048
            win_length=hp.win_length, #1200
            fmin=80,
            fmax=7600
        )
        
        scaler.partial_fit(mel)
        utt2mel[utt_id] = mel
    print(len(wav_files)-len(utt2mel), "utterances ignored")
    
    for utt_id, mel in utt2mel.items():
        mel = scaler.transform(mel)
        np.save(os.path.join(dumpdir, f"{utt_id}-feats.npy"), mel.astype(np.float32), allow_pickle=False)

# +
mel2len = {}
for wav in tqdm(glob.glob(wav_dir + '/*.wav')):
    if get_duration(wav) > 20: continue
    audio, sample_rate = soundfile.read(wav)
    utt_id = os.path.basename(wav).replace('.wav', '')
    mel = logmelfilterbank(
        audio,
        sampling_rate=sample_rate,
        hop_size=300, #300
        fft_size=2048, #2048
        win_length=1200, #1200
        fmin=80,
        fmax=7600
    )
    input_length = len(utt2text[utt_id]) + mel.shape[0]
    mel2len[utt_id] = input_length
    
#print(len(wav_files)-len(utt2mel), "utterances ignored")
# -

with open(os.path.join('utt2shape'), 'w', encoding='utf-8') as f:
    for key, val in mel2len.items():
        f.write(key+' '+str(val)+'\n')

len(mel2len)

# +
wav_dir = 'data/RUSLAN_24k'
dumpdir = "data/RUSLAN_dump"
df = pd.read_csv('data/metadata_RUSLAN_22200.csv', sep='|', header=None, names=['id', 'value'])

def process_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

df.value = df.value.apply(process_text)
utt2text = dict(zip(df.id, df.value))

# -

utt2text

if __name__ == "__main__":
    main()        
