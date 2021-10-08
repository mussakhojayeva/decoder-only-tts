import torch
from torch import nn
import torch.nn.functional as F
from pdb import set_trace
class Linear(nn.Linear):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias=True,
                 w_init_gain='linear'):
        super(Linear, self).__init__(in_dim,
                                     out_dim,
                                     bias)
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain(w_init_gain))


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = Linear(d_model, dim_feedforward, w_init_gain=activation)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt2, dec_align = self.self_attn(tgt,
                                         tgt,
                                         tgt,
                                         attn_mask=tgt_mask,
                                         key_padding_mask=tgt_key_padding_mask)
        
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        
        return tgt, dec_align


