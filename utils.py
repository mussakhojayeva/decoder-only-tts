import numpy as np
import librosa
import os, copy
from scipy import signal
import pdb
import matplotlib.pyplot as plt
from collections import defaultdict

import torch as t
import torch.nn.functional as F

import hparams as hp

def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.
    Args:
      fpath: A string. The full path of a sound file.
    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.num_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def generate_square_subsequent_mask(sz, text_seq_len):
    mask = t.ones((sz, sz)).cuda().triu_(1).bool()
    ## remove masks for text
    for i in range(text_seq_len):
        mask[i][:text_seq_len] = False
    return mask


def get_mask_from_lengths(lengths):
    max_len = t.max(lengths).item()
    ids = lengths.new_tensor(t.arange(0, max_len)).to(lengths.device)
    mask = (lengths.unsqueeze(1) <= ids)
    return mask

def plot_melspec(target, melspec, melspec_post, mel_lengths):
    fig, axes = plt.subplots(3, 1, figsize=(20,30))
    T = mel_lengths[0]
    
    target = target.cpu()
    melspec = melspec.cpu()
    melspec_post = melspec_post.cpu()

    axes[0].imshow(target[0][:T,:],
                   origin='lower',
                   aspect='auto')

    axes[1].imshow(melspec[0][:T,:],
                   origin='lower',
                   aspect='auto')

    axes[2].imshow(melspec_post[0][:T,:],
                   origin='lower',
                   aspect='auto')

    return fig

def plot_gate(gate_out):
    gate_out = gate_out.cpu()
    fig = plt.figure(figsize=(10,5))
    plt.plot(t.sigmoid(gate_out[0]))
    return fig

def plot_alignments(alignments, text_lengths, mel_lengths):
    alignments = alignments.cpu()
    fig, axes = plt.subplots(hp.n_layers, 1, figsize=(5,5*hp.n_layers))
    len_mel = mel_lengths[-1]
    len_text = text_lengths[-1]
    n_layers = alignments.size(1)
    for layer in range(n_layers):
        align = alignments[-1, layer].contiguous()
        axes[layer].imshow(align[:(len_text+len_mel), :(len_text+len_mel)], aspect='auto', origin='lower')

    return fig


