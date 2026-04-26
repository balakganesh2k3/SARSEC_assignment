import torch
import torch.nn as nn
from module import Feedforw, MultiheadAttention

class Sasrec_bl(nn.Module):
    def __init__(self, hid_size, heads_num, dropout_rate):
        super(Sasrec_bl, self).__init__()
        self.norm1 = nn.LayerNorm(hid_size)
        self.attention = MultiheadAttention(hid_size, heads_num, dropout_rate)
        self.norm2 = nn.LayerNorm(hid_size)
        self.ff = Feedforw(hid_size, dropout_rate)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
    
class SASrec(nn.Module):
    def __init__(self, num_items, hid_size, head_nums, block_nums, maxlen, dropout_rate):
        super(SASrec, self).__init__()
        self.hid_size  = hid_size
        self.head_nums = head_nums
        self.block_nums = block_nums
        self.maxlen = maxlen
        self.dropout_rate = dropout_rate
        self.num_items = num_items
        self.item_emb = nn.Embedding(num_items+1, hid_size, padding_idx = 0)
        self.pos_emb = nn.Embedding(maxlen, hid_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList([Sasrec_bl(hid_size, head_nums, dropout_rate) for i in range (block_nums)])
        self.norm = nn.LayerNorm(hid_size)

    def forward(self, seq):
        L = seq.shape[1]
        x = self.item_emb(seq) * (self.hid_size ** 0.5)
        positions = torch.arange(L, device=seq.device)
        pos_emb = self.pos_emb(positions).unsqueeze(0)
        x = x + pos_emb
        x = self.dropout(x)
        pad_mask = (seq != 0).unsqueeze(-1).float()   
        x = x * pad_mask                               
        for block in self.blocks:
            x = block(x)
            x = x * pad_mask                          
        x = self.norm(x)
        return x
    
    def predict(self, seq, item_ids):
        h = self.forward(seq)
        h = h[:, -1, :]
        item_vecs = self.item_emb(item_ids)
        scores = h @ item_vecs.T
        return scores