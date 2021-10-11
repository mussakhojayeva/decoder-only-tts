import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os, re, glob, pdb
import hparams as hp
import torch


def collate_fn(batch):
    
    batch_size = len(batch)
    max_text_len = max([x[0].size(0) for x in batch])
    max_mel_len = max([x[1].size(0) for x in batch])
    num_mels = batch[0][1].size(1)
    
    text_padded = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    mel_padded = torch.zeros(batch_size, max_mel_len, num_mels)
    stop_padded = torch.zeros(len(batch), max_mel_len)
    text_lengths, mel_lengths = torch.zeros(batch_size, dtype=torch.long), torch.zeros(batch_size, dtype=torch.long)
    
    for i, (text, mel) in enumerate(batch):
        text_padded[i, :text.size(0)] = text
        text_lengths[i] = text.size(0)
        mel_padded[i, :mel.size(0), :] = mel
        mel_lengths[i] = mel.size(0)
        stop_padded[i, mel.size(0)-1:] = 1
     
    return text_padded, text_lengths, mel_padded, mel_lengths, stop_padded



class PrepareDataset(Dataset):
    """RUSLAN"""
    def __init__(self, csv_file, wav_dir):
            """
            Args:
                csv_file (string): Path to the csv file with text.
                wav_dir (string): Directory with all the wavs.
            """
            self.dump_dir = wav_dir
            self.unk = "<unk>"
            df = pd.read_csv(csv_file, sep='|', header=None)
            wav_files = [os.path.basename(x.replace('-feats.npy', '')) for x in glob.glob(self.dump_dir + '/*')]
            self.csv_file = df[df.iloc[:, 0].isin(wav_files)]

            self.token_list = {char.lower() for utt in list(self.csv_file.iloc[:, 1]) for char in utt}
            self.token_list.add(self.unk)
            self.token2id: Dict[str, int] = {}
            self.token2id = {t:i for i, t in enumerate(self.token_list)}

    def __len__(self):
        return len(self.csv_file)
    
    def whitespace_clean(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def tokenize(self, text):
        text = self.whitespace_clean(text).lower()
        tokens = []
        for token in text:
            if token in self.token2id:
                token_id = self.token2id[token]
            else: token_id = self.token2id[self.unk]
            tokens.append(token_id) 
        return tokens
    
    def __getitem__(self, idx):
        
        mel_path = os.path.join(self.dump_dir, self.csv_file.iloc[idx, 0]) + '-feats.npy'
        text = self.csv_file.iloc[idx, 1]
        tokenized_text = self.tokenize(text)
        mel = np.load(mel_path)
        
        return torch.tensor(tokenized_text), torch.tensor(mel, dtype=torch.float)
