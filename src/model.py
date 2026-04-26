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
        # Pre-norm attention with residual connection
        x = x + self.attention(self.norm1(x))
        # Pre-norm feed-forward with residual connection
        x = x + self.ff(self.norm2(x))
        return x
<<<<<<< HEAD
    
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
=======


class SASRec(nn.Module):
    def __init__(self, num_items, hidden_size, head_nums, block_nums, maxlen, dropout_rate):
        super(SASRec, self).__init__()
        self.hidden_size  = hidden_size
        self.head_nums    = head_nums
        self.block_nums   = block_nums
        self.maxlen       = maxlen
        self.dropout_rate = dropout_rate
        self.num_items    = num_items

        # +1 for padding index 0, which is reserved and not a real item
        self.item_emb = nn.Embedding(num_items+1, hidden_size, padding_idx=0)
        # One positional embedding per sequence position
        self.pos_emb  = nn.Embedding(maxlen, hidden_size)
        self.dropout  = nn.Dropout(dropout_rate)
        self.blocks   = nn.ModuleList([SASRec_Block(hidden_size, head_nums, dropout_rate)
                                       for i in range(block_nums)])
        # Final layer norm before prediction
        self.norm     = nn.LayerNorm(hidden_size)

    def forward(self, seq):
        L = seq.shape[1]

        # Scale item embeddings by sqrt(hidden_size), as in the original transformer
        x = self.item_emb(seq) * (self.hidden_size ** 0.5)

        # Add positional embeddings to encode order in the sequence
>>>>>>> 171b918b490aacc1a9e0eb188005005156bfda95
        positions = torch.arange(L, device=seq.device)
        pos_emb = self.pos_emb(positions).unsqueeze(0)  # [1, L, hidden_size]
        x = x + pos_emb
        x = self.dropout(x)
<<<<<<< HEAD
        pad_mask = (seq != 0).unsqueeze(-1).float()   
        x = x * pad_mask                               
        for block in self.blocks:
            x = block(x)
            x = x * pad_mask                          
=======

        # Build a mask to suppress padded positions (item_id=0) throughout the blocks
        pad_mask = (seq != 0).unsqueeze(-1).float()  # [B, L, 1]
        x = x * pad_mask

        for block in self.blocks:
            x = block(x)
            x = x * pad_mask  # reapply after each block to prevent padding from leaking

>>>>>>> 171b918b490aacc1a9e0eb188005005156bfda95
        x = self.norm(x)
        return x  # [B, L, hidden_size]

    def predict(self, seq, item_ids):
        h = self.forward(seq)

        # Take only the last position as the user's current state
        h = h[:, -1, :]  # [B, hidden_size]

        # Score each candidate item via dot product with its embedding
        item_vecs = self.item_emb(item_ids)   # [num_items, hidden_size]
        scores = h @ item_vecs.T              # [B, num_items]
        return scores