"""TweetBiDAF: an extractive QA model for tweets, trained from scratch.

  1. Embedding: word embedding ⊕ char-CNN over the *raw* token ⊕ token flags
     -> 2-layer highway                                    (A1: use_char_cnn, A2: use_token_flags)
  2. Shared BiLSTM contextual encoder for tweet (H) and question (U)
  3. Trilinear similarity S_ij = w^T [h_i; u_j; h_i ∘ u_j]
     C2Q = softmax_j(S) U,  Q2C = softmax_i(max_j S) H
     G = [h; c2q; h ∘ c2q; h ∘ q2c]                        (A4: biattn_mode "c2q" drops Q2C)
  4. Gated multi-head self-attention over the projected G,
     residual + LayerNorm                                   (A3: use_self_attn)
  5. 2-layer modelling BiLSTM, start head, 1 more BiLSTM layer, end head,
     padding masks, best (i, j) with i <= j < i + L

The design follows BiDAF (Seo et al., 2017) with gated self-attention in the
style of R-Net (Wang et al., 2017). The tweet-specific parts are written for
this project: a char-CNN on raw tokens with a normalised word vocabulary,
token flags (including the byline flag), soft silver-span training, and
self-attention for long-range links between the tweet body and its byline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG = -1e30

DEFAULT_MODEL = dict(
    word_dim=100, char_dim=32, char_filters=50, char_widths=(3, 4, 5), n_flags=8,
    hidden=128, dropout=0.2, attn_heads=4, max_span_len=15,
    use_char_cnn=True, use_token_flags=True, use_self_attn=True, biattn_mode="both",
)


class CharCNN(nn.Module):
    def __init__(self, n_chars, char_dim, filters, widths):
        super().__init__()
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(char_dim, filters, w, padding=w // 2) for w in widths)
        self.out_dim = filters * len(widths)

    def forward(self, chars):                      # (B, T, W)
        B, T, W = chars.shape
        x = self.emb(chars.view(B * T, W)).transpose(1, 2)            # (BT, c, W)
        cmask = (chars.view(B * T, 1, W) != 0)
        outs = []
        for conv in self.convs:
            y = F.relu(conv(x))[:, :, :W]
            y = y.masked_fill(~cmask, 0.0)          # ReLU output >= 0, so masked max-pool is safe
            outs.append(y.max(dim=2).values)
        return torch.cat(outs, -1).view(B, T, -1)


class Highway(nn.Module):
    def __init__(self, dim, n_layers=2):
        super().__init__()
        self.lin = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))
        self.gate = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_layers))

    def forward(self, x):
        for lin, gate in zip(self.lin, self.gate):
            g = torch.sigmoid(gate(x))
            x = g * F.relu(lin(x)) + (1 - g) * x
        return x


def reverse_padded(x, mask):
    """Reverse each sequence within its own length; padding stays at the end."""
    T = x.size(1)
    lengths = mask.sum(1, keepdim=True)                                  # (B, 1)
    t = torch.arange(T, device=x.device).unsqueeze(0)                    # (1, T)
    idx = torch.where(t < lengths, lengths - 1 - t, t)                   # (B, T)
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))


class BiLSTM(nn.Module):
    """BiLSTM over right-padded batches, padding-exact without packing.

    The forward LSTM reads valid tokens before any padding, so its outputs at
    valid positions are exact. The backward direction runs a forward LSTM over
    each sequence reversed *within its length*, then un-reverses the output.
    The result equals a packed BiLSTM, but runs ~4x faster on MPS.
    """

    def __init__(self, in_dim, hidden, layers=1, dropout=0.0):
        super().__init__()
        self.fw = nn.ModuleList(nn.LSTM(in_dim if l == 0 else 2 * hidden, hidden, batch_first=True)
                                for l in range(layers))
        self.bw = nn.ModuleList(nn.LSTM(in_dim if l == 0 else 2 * hidden, hidden, batch_first=True)
                                for l in range(layers))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        for l, (fw, bw) in enumerate(zip(self.fw, self.bw)):
            if l > 0:
                x = self.drop(x)
            f = fw(x)[0]
            b = reverse_padded(bw(reverse_padded(x, mask))[0], mask)
            x = torch.cat([f, b], -1) * mask.unsqueeze(-1)
        return x


class GatedSelfAttention(nn.Module):
    """Multi-head self-attention, then a sigmoid gate on [x; attended], then residual + LayerNorm."""

    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(2 * dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        attended, w = self.mha(x, x, x, key_padding_mask=~mask,
                               need_weights=True, average_attn_weights=True)
        g = torch.sigmoid(self.gate(torch.cat([x, attended], -1)))
        return self.norm(x + self.drop(g * attended)), w


class TweetBiDAF(nn.Module):
    def __init__(self, n_words, n_chars, cfg=None, word_vectors=None):
        super().__init__()
        self.cfg = cfg = {**DEFAULT_MODEL, **(cfg or {})}
        H, p = cfg["hidden"], cfg["dropout"]
        self.word_emb = nn.Embedding(n_words, cfg["word_dim"], padding_idx=0)
        if word_vectors is not None:
            self.word_emb.weight.data.copy_(word_vectors)
        emb_dim = cfg["word_dim"]
        if cfg["use_char_cnn"]:
            self.char_cnn = CharCNN(n_chars, cfg["char_dim"], cfg["char_filters"], cfg["char_widths"])
            emb_dim += self.char_cnn.out_dim
        if cfg["use_token_flags"]:
            emb_dim += cfg["n_flags"]
        self.highway = Highway(emb_dim, 2)
        self.drop = nn.Dropout(p)
        self.encoder = BiLSTM(emb_dim, H)
        self.w_sim = nn.Linear(6 * H, 1)           # trilinear: w^T [h; u; h∘u]
        g_dim = 8 * H if cfg["biattn_mode"] == "both" else 6 * H
        self.g_proj = nn.Linear(g_dim, 2 * H)
        if cfg["use_self_attn"]:
            self.self_attn = GatedSelfAttention(2 * H, cfg["attn_heads"], p)
        self.model1 = BiLSTM(2 * H, H, layers=2, dropout=p)
        self.model2 = BiLSTM(2 * H, H)
        self.start_head = nn.Linear(4 * H, 1)
        self.end_head = nn.Linear(4 * H, 1)

    # ---------------------------------------------------------------- layers
    def embed(self, words, chars, flags):
        parts = [self.word_emb(words)]
        if self.cfg["use_char_cnn"]:
            parts.append(self.char_cnn(chars))
        if self.cfg["use_token_flags"]:
            parts.append(flags)
        return self.highway(self.drop(torch.cat(parts, -1)))

    def similarity(self, h, u):
        """S_ij = w_h·h_i + w_u·u_j + w_hu·(h_i∘u_j), computed without materialising [h; u; h∘u]."""
        d = h.size(-1)
        w = self.w_sim.weight.squeeze(0)
        wh, wu, whu = w[:d], w[d:2 * d], w[2 * d:]
        return (h @ wh).unsqueeze(2) + (u @ wu).unsqueeze(1) + (h * whu) @ u.transpose(1, 2) + self.w_sim.bias

    def forward(self, batch, return_attn=False):
        tm, qm = batch["t_mask"], batch["q_mask"]
        h = self.drop(self.encoder(self.embed(batch["t_words"], batch["t_chars"], batch["t_flags"]), tm))
        u = self.drop(self.encoder(self.embed(batch["q_words"], batch["q_chars"], batch["q_flags"]), qm))

        S = self.similarity(h, u)                                        # (B, T, J)
        a = F.softmax(S.masked_fill(~qm.unsqueeze(1), NEG), dim=-1)      # C2Q weights
        c2q = a @ u
        parts = [h, c2q, h * c2q]
        b = None
        if self.cfg["biattn_mode"] == "both":
            smax = S.masked_fill(~qm.unsqueeze(1), NEG).max(-1).values    # (B, T)
            b = F.softmax(smax.masked_fill(~tm, NEG), dim=-1)            # Q2C weights
            q2c = (b.unsqueeze(1) @ h).expand_as(h)
            parts.append(h * q2c)
        x = self.drop(F.relu(self.g_proj(torch.cat(parts, -1))))

        sa = None
        if self.cfg["use_self_attn"]:
            x, sa = self.self_attn(x, tm)

        m1 = self.drop(self.model1(x, tm))
        m2 = self.drop(self.model2(m1, tm))
        start = self.start_head(torch.cat([x, m1], -1)).squeeze(-1).masked_fill(~tm, NEG)
        end = self.end_head(torch.cat([x, m2], -1)).squeeze(-1).masked_fill(~tm, NEG)
        out = dict(start_logits=start, end_logits=end)
        if return_attn:
            out.update(S=S, c2q=a, q2c=b, self_attn=sa)
        return out


# ------------------------------------------------------------------ loss / decode

def span_loss(out, batch, soft=True):
    ls = F.log_softmax(out["start_logits"], -1)
    le = F.log_softmax(out["end_logits"], -1)
    if soft:
        # soft cross-entropy against the marginal start/end distributions of the soft silver set
        return -(batch["soft_start"] * ls).sum(-1).mean() - (batch["soft_end"] * le).sum(-1).mean()
    return F.nll_loss(ls, batch["y_start"]) + F.nll_loss(le, batch["y_end"])


@torch.no_grad()
def decode(start_logits, end_logits, max_len=15):
    """argmax over i <= j < i + max_len of log p_s(i) + log p_e(j). Returns (B, 2) indices and scores."""
    ls = F.log_softmax(start_logits, -1)
    le = F.log_softmax(end_logits, -1)
    T = ls.size(1)
    scores = ls.unsqueeze(2) + le.unsqueeze(1)                           # (B, T, T)
    idx = torch.arange(T, device=ls.device)
    band = (idx.unsqueeze(0) >= idx.unsqueeze(1)) & (idx.unsqueeze(0) - idx.unsqueeze(1) < max_len)
    scores = scores.masked_fill(~band, NEG)
    flat = scores.view(scores.size(0), -1)
    best = flat.argmax(-1)
    return torch.stack([best // T, best % T], -1), flat.max(-1).values
