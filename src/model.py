import torch
import torch.nn as nn
from module import FeedForward, MultiheadAttention

class SASRec_Block(nn.Module):
    def __init__(self, hidden_size, heads_num, dropout_rate):
        super(SASRec_Block, self).__init__()
        self.norm1     = nn.LayerNorm(hidden_size)
        self.attention = MultiheadAttention(hidden_size, heads_num, dropout_rate)
        self.norm2     = nn.LayerNorm(hidden_size)
        self.ff        = FeedForward(hidden_size, dropout_rate)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
    
class SASRec(nn.Module):
    def __init__(self, num_items, hidden_size, head_nums, block_nums, maxlen, dropout_rate):
        super(SASRec, self).__init__()
        self.item_emb = nn.Embedding(num_items+1, hidden_size, padding_idx = 0)
        self.pos_emb  = nn.Embedding(maxlen, hidden_size)
        self.dropout  = nn.Dropout(dropout_rate)
        self.blocks   = nn.ModuleList([SASRec_Block(hidden_size, head_nums, dropout_rate)
                                       for i in range (block_nums)])
        self.norm         = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, seq):
        L = seq.shape[1]
        x = self.item_emb(seq) * (self.hidden_size ** 0.5)
        positions = torch.arange(L, device=seq.device)  # [0, 1, 2, ..., L-1]
        pos_emb = self.pos_emb(positions).unsqueeze(0)  # [1, L, hidden_size]
        x = x + pos_emb
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x
    
    def predict(self, seq, item_ids):
        h = self.forward(seq)
        h = h[:, -1, :]
        item_vecs = self.item_emb(item_ids)
        scores = h @ item_vecs.T
        return scores



    