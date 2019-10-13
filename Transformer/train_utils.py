# train_utils.py

import torch
from torch import nn
from torch.autograd import Variable
import copy
import math


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    '''
    Usual Embedding layer with weights multiplied by sqrt(d_model)
    '''
    def __init__(self, config, vocab, w2v_file=None):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, config.embed_size)
        if w2v_file is not None:
            self.lut.weight = nn.Parameter(w2v_file, requires_grad=False)
        self.d_model = config.d_model
        self.inputlayer = nn.Linear(config.embed_size, config.d_model)

    def forward(self, x):
        x = self.inputlayer(self.lut(x))
        return x * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        # torch.cos(position * div_term)
        pe[:, 1::2] = torch.cos(torch.as_tensor(position.numpy() * div_term.unsqueeze(0).numpy()))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)