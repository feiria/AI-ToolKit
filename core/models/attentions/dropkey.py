import torch


def attention_with_dropkey(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask_ratio):
    attn = (q * (q.shape[1] ** -0.5)) @ k.transpose(-2, -1)
    m_r = torch.ones_like(attn) * mask_ratio
    attn = attn + torch.bernoulli(m_r) * -1e12
    attn = attn.softmax(dim=-1)
    x = attn @ v
    return x
