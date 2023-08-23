import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.temperature = d_model ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v):
        n_batches = q.size(0)

        q = self.w_qs(q).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(n_batches, -1, self.n_heads, self.d_k).transpose(1, 2)

        q, attn = self.attention(q, k, v)

        q = q.transpose(1, 2).contiguous().view(n_batches, -1, self.d_model)
        q = self.dropout(self.fc(q))
        return self.layer_norm(q)

class SimpleTransformer(nn.Module):
    def __init__(self, nlayer, d_model=512, n_heads=8):
        super().__init__()
        
        self.layers = nn.ModuleList([MultiHeadAttention(n_heads, d_model) for _ in range(nlayer)])
        # self.layers = nn.ModuleList([MultiHeadAttention(n_heads, d_model)])
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # assuming x is of shape [batch_size, sequence_length, d_model]
        for layer in self.layers:
            x = layer(x, x, x)
        x = x.mean(dim=1)  # taking mean over the sequence length
        x = self.fc(x)
        return x
