import torch
from torch import nn
import numpy as np

from transformer import TransformerDecoderLayer
from utils import get_mask_from_lengths, generate_square_subsequent_mask
import hparams as hp

from pdb import set_trace

class Prenet_D(nn.Module):
    def __init__(self, num_mels, idim):
        super(Prenet_D, self).__init__()
        self.linear1 = nn.Linear(num_mels, idim)
        self.linear2 = nn.Linear(idim, idim)
        self.linear3 = nn.Linear(idim, idim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

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
        dropout_rate=0.5
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
        super().__init__()
        
        self.token2id = token2id
        
        self.padding_idx = 0
        self.n_mels = hp.num_mels
        num_text_tokens = len(token2id) if token2id else 61
        
        self.text_emb = nn.Embedding(num_embeddings=num_text_tokens, embedding_dim=idim, padding_idx=self.padding_idx)
        self.prenet = Prenet_D(self.n_mels, idim)
        
        self.pos_emb = nn.Embedding(num_embeddings=hp.max_seq_len, embedding_dim=idim, padding_idx=self.padding_idx)
                
        self.Decoder = nn.ModuleList([TransformerDecoderLayer(d_model=idim,
                                                              nhead=hp.n_heads,
                                                              dim_feedforward=hp.ff_dim)
                                      for _ in range(hp.n_layers)])

        self.mel_linear = nn.Sequential(
            nn.LayerNorm(idim),
            nn.Linear(idim,  self.n_mels)
        )
        
        self.postnet = Postnet(idim, self.n_mels)

        self.stop_linear = nn.Linear(idim, 1)

    def inference(
        self,
        text: torch.Tensor,
        maxlen=1024
    ):
        
        stop = []
        mel_input = torch.zeros([1, 1, self.n_mels]).cuda()
        for i in range(maxlen):
            mel_pred, stop_tokens = self(text, mel_input)
            stop_token = stop_tokens[:,i]
            stop.append(torch.sigmoid(stop_token)[0,0].item())
            if i < maxlen - 1:
                mel_input = torch.cat([mel_input, mel_pred[:,-1:,:]], dim=1)
            if stop[-1] > 0.5: break
        return mel_pred 
 
    def forward(
        self,
        text: torch.Tensor,
        mel: torch.Tensor,
        text_lengths=None,
        mel_lengths=None
        
    ):

        device = text.device
        
        text_seq_len = text.shape[1] #+ 1 ## one for <bos>
        #text = F.pad(text, (1, 0), value = 0) # <bos> token
        tokens = self.text_emb(text)
        positions = self.pos_emb(torch.arange(1, text_seq_len + 1).to(device))
        
        tokens += positions
       
        
        text_mask = get_mask_from_lengths(text_lengths)
        mel_mask = get_mask_from_lengths(mel_lengths)
        masks = torch.cat((text_mask, mel_mask), dim = 1)
            
        mel_len = mel.shape[1]
        mel_emb = self.prenet(mel)
        positions = self.pos_emb(torch.arange(1, mel_len + 1).to(device))
        mel_emb += positions
        tokens = torch.cat((tokens, mel_emb), dim = 1)
        diag_mask = generate_square_subsequent_mask(mel_len+text_seq_len).to(device)
        
        out = tokens
 
        att_ws = []
        for layer in self.Decoder:
            out, att_w = layer(out, tgt_mask=diag_mask, tgt_key_padding_mask=masks)
            att_ws += [att_w]

        tgt = out[:, text_seq_len:, :]
        
        mel_out = self.mel_linear(tgt)
        mel_out_post = self.postnet(mel_out.transpose(2,1)).transpose(2,1) + mel_out
        stop_tokens = self.stop_linear(tgt)
        att_ws = torch.stack(att_ws, dim=1)

        return mel_out, mel_out_post, stop_tokens, att_ws


