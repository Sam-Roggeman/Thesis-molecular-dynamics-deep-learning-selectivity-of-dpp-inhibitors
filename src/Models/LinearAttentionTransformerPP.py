import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Models.custom_model_template import AbstractNNModel


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, D)
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)
        cos = freqs.cos().unsqueeze(0).unsqueeze(2)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class LinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attn_dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head dimension must be even to use RoPE.")

        self.eps = eps
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    @staticmethod
    def _phi(x: torch.Tensor) -> torch.Tensor:
        # Positive feature map needed for linearized attention.
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q)
        k = self.rope(k)

        q = self._phi(q)
        k = self._phi(k)

        # KV kernel trick: pre-aggregate K^T V once per head for linear complexity in sequence length.
        kv = torch.einsum("bthd,bthm->bhdm", k, v)
        k_sum = k.sum(dim=1)
        z = 1.0 / (torch.einsum("bthd,bhd->bth", q, k_sum) + self.eps)
        out = torch.einsum("bthd,bhdm,bth->bthm", q, kv, z)

        out = out.reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)
        return self.attn_dropout(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * ffn_dim)
        self.out_proj = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gated = self.in_proj(x)
        x_val, x_gate = x_gated.chunk(2, dim=-1)
        x = F.silu(x_gate) * x_val
        x = self.out_proj(x)
        return self.dropout(x)


class TransformerPPBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        attn_dropout: float,
        ffn_dropout: float,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = LinearAttention(
            d_model=d_model,
            n_heads=n_heads,
            attn_dropout=attn_dropout,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model=d_model, ffn_dim=ffn_dim, dropout=ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LinearAttentionTransformerPP(AbstractNNModel):
    def __init__(
        self,
        input_dim: int = 3,
        num_classes: int = 5,
        seq_len: int = 168 * 168,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        ffn_dim = d_model * ffn_multiplier

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.in_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerPPBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    attn_dropout=dropout,
                    ffn_dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected sequence input with shape (B, T, C), got {tuple(inputs.shape)}"
            )

        x = self.input_proj(inputs)
        x = self.in_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.head(x)

    def input_shape(self):
        return self.seq_len, self.input_dim
