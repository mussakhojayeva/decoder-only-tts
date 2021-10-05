from math import log2, sqrt, log
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from transformer import TransformerDecoderLayer
from pdb import set_trace

from utils import get_mask_from_lengths

import hparams as hp


class PositionalEncoding(nn.Module):
    ## sinusoidal 
    ## to-do: add scaled
    def __init__(self, input_size, max_len=5000):
        super().__init__()
        self.max_len = max_len
        pe = torch.zeros(self.max_len, input_size, requires_grad=False)
        positions = torch.arange(0, self.max_len).unsqueeze(1).float()
        denominator = torch.exp(
            torch.arange(0, input_size, 2).float()
            * -(log(10000.0) / input_size)
        )

        pe[:, 0::2] = torch.sin(positions * denominator)
        pe[:, 1::2] = torch.cos(positions * denominator)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)].clone().detach()


class DecoderTTS(nn.Module):
    def __init__(
        self,
        *,
        idim,
        token2id
    ):
        super().__init__()
        
        self.token2id = token2id
        
        self.padding_idx = 0
        self.n_mels = hp.num_mels
        num_text_tokens = len(token2id)
        
        self.text_emb = nn.Embedding(num_embeddings=num_text_tokens, embedding_dim=idim, padding_idx=self.padding_idx)
        self.mel_emb = nn.Linear( self.n_mels, idim, bias=True)
        
        self.text_pos_emb = PositionalEncoding(idim) # + a token for beg. of seq
        self.mel_pos_emb = PositionalEncoding(idim)
                
        self.Decoder = nn.ModuleList([TransformerDecoderLayer(d_model=idim,
                                                              nhead=hp.n_heads,
                                                              dim_feedforward=hp.ff_dim)
                                      for _ in range(hp.n_layers)])
        
        self.mel_linear = nn.Sequential(
            nn.LayerNorm(idim),
            nn.Linear(idim,  self.n_mels)
        )
        
        self.stop_linear = nn.Linear(idim, 1)

    def inference(
        self,
        text: torch.Tensor,
        maxlen=1024
    ):
        set_trace()
        stop = []
        mel_input = t.zeros([1, 1, self.num_mels]).cuda()
        for i in range(max_len):
            mel_pred, stop_tokens = self(text, mel_input)
            stop_token = self.Stop(mel_pred[:,i])
            stop.append(torch.sigmoid(stop_token)[0,0].item())
            if i < max_len-1:
                mel_input = t.cat([mel_input, mel_pred[:,-1:,:]], dim=1)
            if stop[-1] > 0.5: break
        return mel_pred 
 
    def forward(
        self,
        text: torch.Tensor,
        mel: torch.Tensor,
        text_lengths: torch.Tensor,
        mel_lengths: torch.Tensor,
        max_text_len: int,
        max_mel_len: int
        
    ):
        #text = text[:,:text_lengths.max().item()]
        #mel = mel[:,:mel_lengths.max().item(),:]


        #text = F.pad(text, (1, 0), value = 0) # <bos> token
        tokens = self.text_emb(text)
        tokens += self.text_pos_emb(tokens)
        text_seq_len = tokens.shape[1] #+ 1 ## one for <bos>
        
        text_mask = get_mask_from_lengths(text_lengths, max_text_len)
        mel_mask = get_mask_from_lengths(mel_lengths, max_mel_len)

        masks = torch.cat((text_mask, mel_mask), dim = 1)
        if mel.nelement() > 0:
            mel_len = mel.shape[1]
            mel_emb = self.mel_emb(mel)
            mel_emb += self.mel_pos_emb(mel_emb)

            tokens = torch.cat((tokens, mel_emb), dim = 1)
            #feat_seq_len = mel_len
        out = tokens
        for layer in self.Decoder:
            out = layer(out, tgt_key_padding_mask=masks)
        
        #out[:, text_seq_len:, :]
        mel_pred = self.mel_linear(out[:, text_seq_len:, :])
        stop_tokens = self.stop_linear(out[:, text_seq_len:, :])

        return mel_pred, stop_tokens


