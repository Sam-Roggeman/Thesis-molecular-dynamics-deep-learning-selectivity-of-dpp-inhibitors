import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Rotary embeddings
# ------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


# ------------------------------------------------------------
# RMSNorm (better than LayerNorm for transformers++)
# ------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.scale


# ------------------------------------------------------------
# SwiGLU FFN
# ------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_mult=4):
        super().__init__()

        hidden = int(dim * hidden_mult)

        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.out(F.silu(self.w1(x)) * self.w2(x))


# ------------------------------------------------------------
# FAVOR / Linear Attention
#
# Complexity:
#   O(N d^2) instead of O(N^2)
#
# Uses positive kernel feature map:
#   phi(x) = elu(x) + 1
#
# Attention:
#   softmax(QK^T)V
# approximated as:
#   phi(Q)(phi(K)^T V)
# ------------------------------------------------------------

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, head_dim=32):
        super().__init__()

        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim

        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.out = nn.Linear(inner, dim, bias=False)

        self.scale = head_dim ** -0.5

        self.rotary = RotaryEmbedding(head_dim)

    @staticmethod
    def phi(x):
        return F.elu(x) + 1.0

    def forward(self, x):
        """
        x: (B, N, D)
        """

        B, N, _ = x.shape

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.heads, self.head_dim)
        k = k.view(B, N, self.heads, self.head_dim)
        v = v.view(B, N, self.heads, self.head_dim)

        # rotary
        cos, sin = self.rotary(N, x.device)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        q = self.phi(q)
        k = self.phi(k)

        # ----------------------------------------------------
        # KV compression
        #
        # kv:
        #   (B, H, D, D)
        #
        # instead of materializing NxN attention
        # ----------------------------------------------------

        kv = torch.einsum(
            "bnhd,bnhe->bhde",
            k,
            v
        )

        # normalizer
        z = 1.0 / (
            torch.einsum(
                "bnhd,bhd->bnh",
                q,
                k.sum(dim=1)
            ) + 1e-6
        )

        out = torch.einsum(
            "bnhd,bhde,bnh->bnhe",
            q,
            kv,
            z
        )

        out = out.reshape(B, N, -1)

        return self.out(out)


# ------------------------------------------------------------
# Transformer++ Block
#
# Improvements:
#   - PreNorm
#   - RMSNorm
#   - SwiGLU
#   - Rotary
#   - Residual scaling
#   - Linear attention
# ------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        head_dim=32,
        ff_mult=4,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.attn = LinearAttention(
            dim=dim,
            heads=heads,
            head_dim=head_dim,
        )

        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, hidden_mult=ff_mult)

        # residual scaling stabilizes deep training
        self.res_scale = 0.5

    def forward(self, x):
        x = x + self.res_scale * self.attn(self.norm1(x))
        x = x + self.res_scale * self.ff(self.norm2(x))
        return x


# ------------------------------------------------------------
# Atom embedding
#
# Input:
#   atom_idx : (B, N)
#   xyz      : (B, N, 3)
#
# Since atom indices are arbitrary and NOT positional:
#   - atom embedding handles atom identity
#   - xyz projected continuously
#   - rotary still provides sequence structure
#
# IMPORTANT:
# If permutation invariance matters strongly,
# consider replacing rotary with pairwise geometric bias.
# ------------------------------------------------------------

class AtomEmbedding(nn.Module):
    def __init__(self, num_atoms, dim):
        super().__init__()

        self.atom_emb = nn.Embedding(num_atoms, dim)
        self.xyz_proj = nn.Linear(3, dim)

        self.mix = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, atom_idx, xyz):

        atom_feat = self.atom_emb(atom_idx)
        xyz_feat = self.xyz_proj(xyz)

        x = torch.cat([atom_feat, xyz_feat], dim=-1)

        return self.mix(x)


# ------------------------------------------------------------
# Final Model
#
# Small by default for long seqs (~25k)
# ------------------------------------------------------------

class LongSequenceAtomTransformer(nn.Module):
    def __init__(
        self,
        num_atoms = 168*168,  # max number of pixels (atoms) in the input
        d_model=64,
        depth=4,
        heads=4,
        head_dim=16,
        ff_mult=4,
        num_classes=5,
        dropout=0.1,
    ):
        super().__init__()

        self.embed = AtomEmbedding(
            num_atoms=num_atoms,
            dim=d_model,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                heads=heads,
                head_dim=head_dim,
                ff_mult=ff_mult,
            )
            for _ in range(depth)
        ])

        self.norm = RMSNorm(d_model)

        # mean pooling head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(0.5),   
            nn.Linear(d_model, num_classes),
        )

    def forward(self, atom_idx, xyz):
        """
        atom_idx : (B, N)
        xyz      : (B, N, 3)
        """

        x = self.embed(atom_idx, xyz)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # mean pooling for classification
        x = x.mean(dim=1)

        return self.head(x)

