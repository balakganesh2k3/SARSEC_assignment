import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size  )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate):
        super(MultiheadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, head_num, dropout = dropout_rate)
        self.dropout    = nn.Dropout(dropout_rate)

    def forward(self, x):
        L = x.shape[1]
        mask = torch.triu(torch.ones(L,L), diagonal=1).bool().to(x.device)
        x = x.transpose(0, 1)
        x, attention_weights = self.attention(x, x, x, attn_mask = mask)
        x = x.transpose(0,1)
        x = self.dropout(x)
        return x
