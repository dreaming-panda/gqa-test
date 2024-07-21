import torch
from utils import gqa_custom, gqa_merge
from einops import rearrange
gqa_attn = torch.ops.mylib.gqa_custom

def group_query_attention(q, k_cache, v_cache, k, v, cache_seqlens, insert_indices):

    B, T, H_q, D = q.size()
    H_k = k.size(2)
    rep = H_q // H_k
    q_reshaped = q.view(B, T, H_k, rep, D).transpose(2, 3).contiguous().view(B, T*rep, H_k, D).contiguous()
    
    y_past, lse_past, y_new, lse_new = gqa_attn(q, q_reshaped, k_cache, v_cache, k, v, cache_seqlens)

    y_past = y_past.view(B, T, rep, H_k, D).transpose(2, 3).contiguous().view(B, T, H_q, D)
    # lse_past = rearrange(lse_past, 'b h (t r) -> b t (h r) 1', r=rep).contiguous()
    
    #lse_new = lse_new.unsqueeze(-1).transpose(1, 2)
    
    # sumexp_past = torch.exp(lse_past.float())
    # sumexp_new = torch.exp(lse_new.float())

    # sumexp_total = sumexp_past + sumexp_new
    # y = (y_past * sumexp_past + y_new * sumexp_new) / sumexp_total
    # k_cache.scatter_(1, insert_indices, k)
    # v_cache.scatter_(1, insert_indices, v)   

    return y_past.to(k.dtype)
    