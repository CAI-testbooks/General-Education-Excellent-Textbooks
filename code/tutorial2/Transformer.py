'''
Author: Chengyu Zheng
Date: 2024-11-17
Description: 
'''
import math

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

## ScaledDotProductAttention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    def forward(self, Q, K, V, attn_mask):

##  MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
    def forward(self, Q, K, V, attn_mask):

##  get_attn_subsequent_mask
def get_attn_subsequent_mask(seq):

##  PositionalEncoding 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pe', pe) 
    def forward(self, x):

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(2).unsqueeze(1) 
    return pad_attn_mask.expand(batch_size, len_q, len_k) 

##  PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()

    def forward(self, inputs):

##  EncoderLayer 
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
    def forward(self, enc_inputs, enc_self_attn_mask):

##  Encoder 
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, n_layers, n_heads, dim_feedforward, dropout):
        super(Encoder, self).__init__()
    def forward(self, enc_inputs):

##  DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(DecoderLayer, self).__init__()
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):

##  Decoder
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, n_layers, n_heads, dim_feedforward, dropout):
        super(Decoder, self).__init__()
    def forward(self, dec_inputs, enc_inputs, enc_outputs): 

##   Transformer
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nheads, dim_feedforward, nlayers, dropout=0.5):
        super(Transformer, self).__init__()
    def forward(self, enc_inputs, dec_inputs):