"""
ITFormer Adapted - For univariate/single-channel time series (b, l, d)
Removes variable-related components, replaces ITAttBlock with cross-attention.
Supports different dimensions for query (D_LLM) and key/value (D_TS).
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import DropPath


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


class SeqAttention(nn.Module):
    """Standard Multi-Head Self-Attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttBlock(nn.Module):
    """Self-Attention + LayerNorm + residual."""
    def __init__(self, dim, num_heads, qkv_bias=False, qk_norm=False,
                 proj_drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn_seq = SeqAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, attn_mask=None):
        x = x + self.drop_path1(self.attn_seq(self.norm1(x), attn_mask))
        return x


class CrossAttention(nn.Module):
    """Cross-Attention: query attends to key_value (memory).

    Supports three independent dimensions:
        query:     (B, N_q, q_dim)   — prefix tokens (D_LLM)
        key_value: (B, N_kv, kv_dim) — memory sequence (D_TS)
        qk_dim:    inner attention dimension (user-defined)

    Q is projected from q_dim → qk_dim, K/V from kv_dim → qk_dim.
    The output is projected from qk_dim back to q_dim.
    """
    def __init__(self, q_dim, kv_dim, qk_dim, num_heads=8, qkv_bias=False, qk_norm=False,
                 attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert qk_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = qk_dim // num_heads

        self.q_proj = nn.Linear(q_dim, qk_dim, bias=qkv_bias)       # q_dim → qk_dim
        self.kv_proj = nn.Linear(kv_dim, qk_dim * 2, bias=qkv_bias) # kv_dim → 2 * qk_dim
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(qk_dim, q_dim)                        # qk_dim → q_dim
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key_value, attn_mask=None):
        B, N_q, _ = query.shape
        N_kv = key_value.shape[1]

        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N_q, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttBlock(nn.Module):
    """Cross-Attention block with norm + residual."""
    def __init__(self, q_dim, kv_dim, qk_dim, num_heads, qkv_bias=False, qk_norm=False,
                 proj_drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_q = norm_layer(q_dim)
        self.norm_kv = norm_layer(kv_dim)
        self.cross_attn = CrossAttention(
            q_dim, kv_dim, qk_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, key_value, attn_mask=None):
        query = query + self.drop_path(
            self.cross_attn(self.norm_q(query), self.norm_kv(key_value), attn_mask)
        )
        return query


class DecoderBasicBlock(nn.Module):
    """Adapted ITFormer Layer.

    Flow:
      1. Self-attention on all tokens (prefix + instruction) in q_dim space
      2. FFN for all tokens
      3. Extract prefix tokens
      4. Cross-attention: prefix (Q, q_dim) ↔ memory (K/V, kv_dim)
      5. FFN for prefix
      6. Concat prefix back with remaining tokens
    """
    def __init__(self, q_dim, kv_dim, qk_dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_norm=False,
                 proj_drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, prefix_num=10):
        super().__init__()
        self.prefix_num = prefix_num

        # Self-attention operates in q_dim space (prefix + instruction are both q_dim)
        self.self_attn = SelfAttBlock(
            dim=q_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path,
            norm_layer=norm_layer,
        )

        # FFN for all tokens (q_dim)
        self.feed_forward_instruct = nn.Sequential(
            norm_layer(q_dim),
            nn.Linear(q_dim, int(q_dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(int(q_dim * mlp_ratio), q_dim),
        )

        # Cross-attention: prefix (q_dim) ↔ memory (kv_dim), inner dim qk_dim
        self.cross_attn = CrossAttBlock(
            q_dim=q_dim, kv_dim=kv_dim, qk_dim=qk_dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, drop_path=drop_path,
            norm_layer=norm_layer,
        )

        # FFN for prefix (q_dim)
        self.feed_forward_prefix = nn.Sequential(
            norm_layer(q_dim),
            nn.Linear(q_dim, int(q_dim * mlp_ratio)),
            act_layer(),
            nn.Dropout(proj_drop),
            nn.Linear(int(q_dim * mlp_ratio), q_dim),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, memory, attn_mask=None):
        # x:      (B, prefix_num + L_x, q_dim)
        # memory: (B, L_m, kv_dim)

        # Self-attention on all tokens
        x = self.self_attn(x, attn_mask)

        # FFN for all tokens
        x = x + self.feed_forward_instruct(x)

        # Extract prefix
        prefix = x[:, :self.prefix_num, :]

        # Cross-attention — prefix (Q) attends to memory (K/V)
        prefix = self.cross_attn(prefix, memory, attn_mask=None)

        # FFN for prefix
        prefix = prefix + self.feed_forward_prefix(prefix)

        # Concat prefix back
        x = torch.cat([prefix, x[:, self.prefix_num:, :]], dim=1)
        return x


class ITFormerAdapted(nn.Module):
    """Adapted ITFormer for input shape (b, l, d) — no variable/channel dimension.

    Supports different dimensions for instruction/prefix tokens (D_LLM)
    and time-series memory (D_TS).

    Args:
        args: config object with fields:
            - it_d_model:  query/instruction dimension (D_LLM)
            - it_d_ts:     time-series memory dimension (D_TS)
            - it_qk_dim:   inner attention dimension for cross-attention
            - it_n_heads:  number of attention heads
            - it_layers:   number of decoder blocks
            - it_dropout:  dropout rate
            - prefix_num:  number of prefix tokens

    Forward:
        x:      (B, L_x, D_LLM) — instruction/query tokens
        memory: (B, L_m, D_TS)  — time series memory (already embedded)

    Returns:
        (B, prefix_num, D_LLM) — prefix token outputs
    """
    def __init__(self, args):
        super().__init__()

        q_dim = args.it_d_model   # D_LLM
        kv_dim = args.it_d_ts     # D_TS
        qk_dim = args.it_qk_dim   # inner attention dimension for cross-attention

        self.prefix_num = args.prefix_num
        self.prefix_token = nn.Parameter(torch.randn(1, args.prefix_num, q_dim))

        # Positional encoding in q_dim space
        self.instruc_pos = SinusoidalPositionalEncoding(q_dim)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBasicBlock(
                q_dim=q_dim,
                kv_dim=kv_dim,
                qk_dim=qk_dim,
                num_heads=args.it_n_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_norm=False,
                proj_drop=args.it_dropout,
                attn_drop=args.it_dropout,
                drop_path=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                prefix_num=args.prefix_num,
            ) for _ in range(args.it_layers)
        ])

        self.norm = nn.LayerNorm(q_dim)

    def forward(self, x, memory, attn_mask=None):
        """
        x:      (B, L_x, D_LLM) — instruction tokens
        memory: (B, L_m, D_TS)  — time series memory
        """
        # Prepend learnable prefix tokens with positional encoding
        prefix = self.prefix_token.expand(x.shape[0], -1, -1)
        prefix = prefix + self.instruc_pos(prefix)
        x = torch.cat([prefix, x], dim=1)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, memory, attn_mask)

        x = self.norm(x)

        # Return only prefix tokens
        return x[:, :self.prefix_num, :]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    class Args:
        def __init__(self):
            self.it_d_model = 512    # D_LLM
            self.it_d_ts = 256       # D_TS (different from D_LLM)
            self.it_qk_dim = 384     # inner attention dimension (independent of both), set it as D_LLM
            self.it_n_heads = 8
            self.it_layers = 4
            self.it_dropout = 0.1
            self.prefix_num = 20

    args = Args()
    model = ITFormerAdapted(args)

    batch_size = 2
    seq_len_x = 20       # instruction token length
    seq_len_mem = 96      # time series memory length
    d_llm = args.it_d_model
    d_ts = args.it_d_ts

    x = torch.randn(batch_size, seq_len_x, d_llm)
    memory = torch.randn(batch_size, seq_len_mem, d_ts)

    output = model(x, memory)
    print(f"Output shape: {output.shape}")  # (2, 10, 512)
    print(f"Total trainable parameters: {count_parameters(model):,}")