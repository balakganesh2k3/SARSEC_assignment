import torch
import torch.nn as nn

class Feedforw(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(Feedforw, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Linear → ReLU → Dropout → Linear → Dropout
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, hid_size, head_num, dropout_rate):
        super(MultiheadAttention, self).__init__()
        # PyTorch's built-in MHA; dropout applied to attention weights internally
        self.attention = nn.MultiheadAttention(hid_size, head_num, dropout = dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, x):
        L = x.shape[1]
        # Causal mask: upper triangle is True, so each position can only attend to itself and the past
        # Shape [L, L] — True entries are blocked (set to -inf before softmax)
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(x.device)
        # nn.MultiheadAttention expects [L, B, hidden_size]
        x = x.transpose(0, 1)
        x, _ = self.attention(x, x, x, attn_mask=mask)  # self-attention: Q = K = V = x
        x = x.transpose(0, 1)  # back to [B, L, hidden_size]
        x = self.dropout(x)
        return x