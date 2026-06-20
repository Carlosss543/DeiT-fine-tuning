import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type, Final
import torch
from timm.layers.attention import resolve_self_attn_mask, maybe_add_mask, use_fused_attn


class MyAttention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            attn_head_dim: Optional[int] = None,
            dim_out: Optional[int] = None,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            scale_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Type[nn.Module]] = None,
            device=None,
            dtype=None,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings.
            num_heads: Number of attention heads.
            attn_head_dim: Dimension of each attention head. If None, computed as dim // num_heads.
            dim_out: Output dimension. If None, same as dim.
            qkv_bias: Whether to use bias in the query, key, value projections.
            qk_norm: Whether to apply normalization to query and key vectors.
            scale_norm: Whether to apply normalization to attention output before projection.
            proj_bias: Whether to use bias in the output projection.
            attn_drop: Dropout rate applied to the attention weights.
            proj_drop: Dropout rate applied after the output projection.
            norm_layer: Normalization layer constructor for QK normalization if enabled.
        """
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        dim_out = dim_out or dim
        head_dim = attn_head_dim
        if head_dim is None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            head_dim = dim // num_heads
        if qk_norm or scale_norm:
            assert norm_layer is not None, 'norm_layer must be provided if qk_norm or scale_norm is True'

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, self.attn_dim * 3, bias=qkv_bias, **dd)
        self.q_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim, **dd) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(self.attn_dim, **dd) if scale_norm else nn.Identity()
        self.proj = nn.Linear(self.attn_dim, dim_out, bias=proj_bias, **dd)
        self.proj_drop = nn.Dropout(proj_drop)

        self.bias_topk = None
        self.register_buffer("last_pruned_ratio", torch.zeros(num_heads), persistent=False)
        self.b = None # linear layer to generate bias for each head, if use_bias is True

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False,) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x).reshape(B, T, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if self.b is not None:
            b = self.b(x)  # (B, T, H)
            b = b.permute(0, 2, 1)  # (B, H, T)

            if self.bias_topk is not None:
                k = int(self.bias_topk * T)
                _, topk_indices = torch.topk(b, k, dim=-1)  # (B, H, K)
                mask = torch.zeros_like(b, dtype=torch.bool).scatter(-1, topk_indices, True)
                b = torch.where(mask, b, float('-inf'))  # (B, H, T)
                self.last_pruned_ratio.fill_(1.0 - (k / T)) # pruning ratio for topk

            attn = attn + b.unsqueeze(2)  # (B, H, T, T) + (B, H, 1, T)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v # (B, H, T, D)

        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.norm(out)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out
