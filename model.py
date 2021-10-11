import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace
import math

from utils import get_mask_from_lengths
import hparams as hp
from transformer import TransformerLayer

class Prenet(nn.Module):
    def __init__(self, num_mels, idim, dropout_rate=0.1):
        super(Prenet, self).__init__()
        self.linear1 = nn.Linear(num_mels, idim)
        self.linear2 = nn.Linear(idim, idim)
        self.linear3 = nn.Linear(idim, idim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.linear3(x)
        return x
    

class Postnet(nn.Module):
    def __init__(
        self,
        idim,
        odim,
        n_layers=5,
        n_chans=512,
        n_filts=5,
        dropout_rate=0.1
    ):
        super(Postnet, self).__init__()
        self.postnet = nn.ModuleList()
        for layer in range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            self.postnet += [
                nn.Sequential(
                    nn.Conv1d(
                        ichans,
                        ochans,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(ochans),
                    nn.Tanh(),
                    nn.Dropout(dropout_rate),
                )
            ]
 
        ichans = n_chans if n_layers != 1 else odim
        self.postnet += [
            nn.Sequential(
                nn.Conv1d(
                    ichans,
                    odim,
                    n_filts,
                    stride=1,
                    padding=(n_filts - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(odim),
                nn.Dropout(dropout_rate),
            )
        ]


    def forward(self, xs):
        for i in range(len(self.postnet)):
            xs = self.postnet[i](xs)
        return xs
    
    
class DecoderTTS(nn.Module):
    def __init__(
        self,
        *,
        idim,
        token2id=None
    ):
        super(DecoderTTS, self).__init__()
        
        self.token2id = token2id
        
        self.padding_idx = 0
        self.n_mels = hp.num_mels
        num_text_tokens = len(token2id) if token2id else 62
        
        self.eos = num_text_tokens-1
        
        self.text_emb = nn.Embedding(num_embeddings=num_text_tokens, 
                                     embedding_dim=idim, 
                                     padding_idx=self.padding_idx)
        #self.mel_emb = nn.Linear(self.n_mels, idim, bias=True)
        self.mel_emb = Prenet(self.n_mels, idim)
        
        self.att_num_buckets = hp.att_num_buckets
        self.relative_attention_bias = nn.Embedding(self.att_num_buckets, 
                                                    hp.n_heads, 
                                                    padding_idx=self.padding_idx)
       
        self.mel_linear = nn.Sequential(
            nn.LayerNorm(idim),
            nn.Linear(idim,  self.n_mels)
        )
        
        self.postnet = Postnet(idim, self.n_mels)

        self.stop_linear = nn.Linear(idim, 1)
        
        self.Decoder = nn.ModuleList([TransformerLayer(dim=idim,
                                                      heads=hp.n_heads, causal=True, stable=False)
                              for _ in range(hp.n_layers)])

    def compute_position_bias(self, x, num_buckets):
        bsz, qlen, klen = x.size(0), x.size(1), x.size(1)
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        
        relative_position = memory_position - context_position
    
        rp_bucket = self.relative_position_bucket(
            relative_position,
            num_buckets=num_buckets
        )
        rp_bucket = rp_bucket.to(x.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1])#.unsqueeze(0)
        values = values.expand((bsz, -1, qlen, klen)).contiguous()
        #values = values.view(-1, qlen, klen)
        return values 
    
    @staticmethod
    def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0 
        n = -relative_position
    
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
    
        max_exact = num_buckets // 2
        is_small = n < max_exact
    
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
    
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret   
    
    def inference(
        self,
        text: torch.Tensor,
        maxlen=1024
    ):
        
        stop = []
        mel_input = torch.zeros([1, 1, self.n_mels]).cuda()
        text = F.pad(text, [0, 1], "constant", self.eos)
        for i in range(maxlen):
            mel_out, mel_out_post, stop_tokens, _ = self(text, mel_input)
            stop_token = stop_tokens[:,i]
            stop.append(torch.sigmoid(stop_token)[0,0].item())
            if i < maxlen - 1:
                mel_input = torch.cat([mel_input, mel_out[:,-1:,:]], dim=1)
            if stop[-1] > 0.5: break
                
        return mel_out_post 
 
    def forward(
        self,
        text: torch.Tensor,
        mel: torch.Tensor,
        text_lengths=None,
        mel_lengths=None
        
    ):
        device = text.device
        
        if torch.is_tensor(text_lengths):
            text = F.pad(text, [0, 1], "constant", self.padding_idx)
            for i, l in enumerate(text_lengths):
                text[i, l] = self.eos
            text_lengths = text_lengths + 1
        
        text_seq_len = text.shape[1]
        tokens = self.text_emb(text) 
        
        if torch.is_tensor(text_lengths) and torch.is_tensor(mel_lengths):
            mel_mask = get_mask_from_lengths(mel_lengths)
            text_mask = get_mask_from_lengths(text_lengths)
            masks = torch.cat((text_mask, mel_mask), dim = 1)
        else: masks = None
            
        mel_emb = self.mel_emb(mel)
        tokens = torch.cat((tokens, mel_emb), dim = 1)
        
        out = tokens
        positions_bias = self.compute_position_bias(out, self.att_num_buckets)
        
        att_ws = []
        for layer in self.Decoder:
            out, att_w = layer(out, tgt_key_padding_mask=masks, positions_bias=positions_bias)
            att_ws += [att_w]
        att_ws = torch.stack(att_ws, dim=1)
                
        stop_tokens = self.stop_linear(out[:, text_seq_len:, :])
        mel_out = self.mel_linear(out[:, text_seq_len:, :])
        post_out = self.postnet(mel_out.transpose(2,1)).transpose(2,1)
        mel_post_out = post_out + mel_out

        return mel_out, mel_post_out, stop_tokens, att_ws

