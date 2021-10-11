import torch
from torch import nn
import torch.nn.functional as F
import pdb
from attention import Attention


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        causal = True,
        heads = 4,
        dim_head = 64,
        ff_dim=2048,
        stable = False,
        dropout=0.1
    ):
        super().__init__()
        
        self.attn = Attention(dim, causal = causal, stable = stable, heads = heads, dim_head = dim_head)

        self.linear1 = nn.Linear(dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, dim)

        self.norm = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_key_padding_mask=None, positions_bias=None):
        tgt2, dec_align = self.attn(tgt, mask=tgt_key_padding_mask, positions_bias=positions_bias)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        
        return tgt, dec_align
        
        
        
     
        
        