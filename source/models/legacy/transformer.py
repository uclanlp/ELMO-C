# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from timeit import default_timer as timer

##########################################################

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, first_layer = None):
        super(Encoder, self).__init__()
        if first_layer:
            self.layers = nn.ModuleList([copy.deepcopy(first_layer)])
            for i in N - 1:
                self.layers.append(copy.deepcopy(layer))
        else:
            self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def get_all_layer_output(self, x, mask):
        output_list = []
        y = x
        for layer in self.layers:
            output_list.append(x)
            x, y = layer.get_all_layer_output(x, mask)
        x = self.norm(x)
        output_list.append(x)
        return torch.stack(output_list, dim = 0) # So we actually returns layer+1 representations
 
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

    def get_all_layer_output(self, x, sublayer):
        y = self.dropout(sublayer(self.norm(x)))
        return x + y, y

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

    def get_all_layer_output(self, x , mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x, y = self.sublayer[1].get_all_layer_output(x, self.feed_forward)
        return x, y

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0     

def subsequent_mask_reverse(size): ## Very important !!!
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=0).astype('uint8')
    return torch.from_numpy(subsequent_mask)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value =             [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous()              .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# ## Positional Encoding                                                                                
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False) # This is to be compatible with the SRL model where we use Pytorch 0.3
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_size = 300, N=6, 
                   d_model=300, d_ff=300, h=4, dropout=0.1, reverse = False):
        super(Transformer, self).__init__()

        c = copy.deepcopy
        #attn_first_layer = MultiHeadedAttention(h, input_size)
        if input_size != d_model:
            self.input_projection = nn.Linear(input_size, d_model)
        else:
            self.input_projection = None

        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.position = PositionalEncoding(d_model, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, c(attn), 
                                 c(ff), dropout), N)

        self.reverse = reverse

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, input, mask, hidden = None):
        if self.input_projection:
            input = self.input_projection(input)
        input = self.position(input)
        return self.encoder(input, self.new_mask(mask, self.reverse))

    def get_all_layer_output(self, input, mask, no_carry = False):
        if self.input_projection:
            input = self.input_projection(input)
        input = self.position(input)

        outputs = self.encoder.get_all_layer_output(input, self.new_mask(mask, self.reverse))
        return outputs
        
    
    def new_mask(self, old_mask, reverse, pad = 0):
        old_mask = old_mask.unsqueeze(-2)
        if reverse:
            tgt_mask = old_mask & Variable(subsequent_mask_reverse(old_mask.size(-1)).type_as(old_mask.data))
        else:
            tgt_mask = old_mask & Variable(subsequent_mask(old_mask.size(-1)).type_as(old_mask.data))
        return tgt_mask