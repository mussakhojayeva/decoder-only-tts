import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
import pdb

def exists(val):
    return val is not None

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def stable_softmax(t, dim = -1, alpha = 32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim = dim, keepdim = True)
    return (t * alpha).softmax(dim = dim)

class Attention(nn.Module):
    def __init__(self, dim, causal = True, heads = 8, dim_head = 64, stable = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.stable = stable
        self.causal = causal

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None, positions_bias=None, need_weights=True):
        b, n, _, h = *x.shape, self.heads
        softmax = torch.softmax if not self.stable else stable_softmax

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        q = q * self.scale

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(mask, mask_value)
            del mask
            
        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j).cuda().triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)
            
        if exists(positions_bias):
            dots += positions_bias


        attn = softmax(dots, dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        
        # average attention weights over heads
        attn = attn.view(b, self.heads, n, n)
        attn = attn.sum(dim=1) / self.heads
            
        return out, attn