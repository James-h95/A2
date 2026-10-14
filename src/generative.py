"""Small pointer-generator for TweetQA, trained from scratch on the provided data."""
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import data as D
from .model import BiLSTM, CharCNN, GatedSelfAttention, Highway, NEG

GEN_SPECIALS = ("<pad>", "<unk>", "<bos>", "<eos>")


class GenVocab:
    def __init__(self, tokens, min_freq=2):
        counts = Counter(tokens)
        self.itos = list(GEN_SPECIALS) + sorted(w for w, n in counts.items()
                                                 if n >= min_freq and w not in GEN_SPECIALS)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def id(self, token):
        return self.stoi.get(token, 1)


def raw_answer_tokens(text):
    return [raw.lower() for raw, _, _, _ in D.tokenize(text)]


def build_gen_vocab(train_rows, min_freq=2):
    return GenVocab((t for r in train_rows for a in r["answers"] for t in raw_answer_tokens(a)), min_freq)


def add_generation_fields(examples, gv, max_answer_len=16):
    """Add per-example extended-vocabulary copy ids and decoder targets."""
    V = len(gv)
    for ex in examples:
        source = [t["raw"].lower() for t in ex["t_tokens"]]
        oovs, oov_to_i = [], {}
        copy_ids = []
        for token in source:
            if token in gv.stoi:
                copy_ids.append(gv.stoi[token])
            else:
                if token not in oov_to_i:
                    oov_to_i[token] = len(oovs); oovs.append(token)
                copy_ids.append(V + oov_to_i[token])
        target_tokens = raw_answer_tokens(ex["answers"][0])[:max_answer_len - 1]
        target_ext = [gv.stoi[t] if t in gv.stoi else V + oov_to_i[t] if t in oov_to_i else 1
                      for t in target_tokens] + [3]
        decoder_in = [2] + [i if i < V else 1 for i in target_ext[:-1]]
        ex.update(copy_ids=copy_ids, source_oovs=oovs, target_tokens=target_tokens,
                  target_ext=target_ext, decoder_in=decoder_in)
    return examples


def collate_generation(examples):
    batch = D.collate(examples)
    B, T = batch["t_words"].shape
    L = max(len(e["target_ext"]) for e in examples)
    copy_ids = torch.zeros(B, T, dtype=torch.long)
    decoder_in = torch.zeros(B, L, dtype=torch.long)
    target_ext = torch.zeros(B, L, dtype=torch.long)
    target_mask = torch.zeros(B, L, dtype=torch.bool)
    for b, e in enumerate(examples):
        copy_ids[b, :len(e["copy_ids"])] = torch.tensor(e["copy_ids"])
        n = len(e["target_ext"])
        decoder_in[b, :n] = torch.tensor(e["decoder_in"])
        target_ext[b, :n] = torch.tensor(e["target_ext"])
        target_mask[b, :n] = True
    batch.update(copy_ids=copy_ids, decoder_in=decoder_in, target_ext=target_ext,
                 target_mask=target_mask,
                 max_oov=max((len(e["source_oovs"]) for e in examples), default=0))
    return batch


class TweetPointerGenerator(nn.Module):
    def __init__(self, n_words, n_chars, gen_vocab_size, cfg, word_vectors=None):
        super().__init__()
        self.cfg, self.gen_vocab_size = cfg, gen_vocab_size
        H, p = cfg["hidden"], cfg["dropout"]
        self.word_emb = nn.Embedding(n_words, cfg["word_dim"], padding_idx=0)
        if word_vectors is not None:
            self.word_emb.weight.data.copy_(word_vectors)
        emb_dim = cfg["word_dim"]
        self.char_cnn = CharCNN(n_chars, cfg["char_dim"], cfg["char_filters"], cfg["char_widths"])
        emb_dim += self.char_cnn.out_dim
        if cfg["use_token_flags"]: emb_dim += cfg["n_flags"]
        self.highway = Highway(emb_dim, 2)
        self.drop = nn.Dropout(p)
        self.encoder = BiLSTM(emb_dim, H)
        self.w_sim = nn.Linear(6 * H, 1)
        self.g_proj = nn.Linear(8 * H, 2 * H)
        self.self_attn = GatedSelfAttention(2 * H, cfg["attn_heads"], p)

        dim = 2 * H
        self.dec_emb = nn.Embedding(gen_vocab_size, cfg.get("decoder_emb", 100), padding_idx=0)
        self.attn_mem = nn.Linear(dim, dim, bias=False)
        self.attn_dec = nn.Linear(dim, dim, bias=False)
        self.attn_v = nn.Linear(dim, 1, bias=False)
        self.decoder = nn.GRUCell(cfg.get("decoder_emb", 100) + dim, dim)
        self.vocab_out = nn.Linear(2 * dim, gen_vocab_size)
        self.p_gen = nn.Linear(2 * dim + cfg.get("decoder_emb", 100), 1)

    def embed(self, words, chars, flags):
        parts = [self.word_emb(words), self.char_cnn(chars)]
        if self.cfg["use_token_flags"]: parts.append(flags)
        return self.highway(self.drop(torch.cat(parts, -1)))

    def encode(self, batch):
        tm, qm = batch["t_mask"], batch["q_mask"]
        h = self.drop(self.encoder(self.embed(batch["t_words"], batch["t_chars"], batch["t_flags"]), tm))
        u = self.drop(self.encoder(self.embed(batch["q_words"], batch["q_chars"], batch["q_flags"]), qm))
        d = h.size(-1); w = self.w_sim.weight.squeeze(0)
        S = ((h @ w[:d]).unsqueeze(2) + (u @ w[d:2*d]).unsqueeze(1)
             + (h * w[2*d:]) @ u.transpose(1, 2) + self.w_sim.bias)
        a = F.softmax(S.masked_fill(~qm.unsqueeze(1), NEG), -1); c2q = a @ u
        smax = S.masked_fill(~qm.unsqueeze(1), NEG).max(-1).values
        b = F.softmax(smax.masked_fill(~tm, NEG), -1)
        q2c = (b.unsqueeze(1) @ h).expand_as(h)
        memory = self.drop(F.relu(self.g_proj(torch.cat([h, c2q, h*c2q, h*q2c], -1))))
        memory, _ = self.self_attn(memory, tm)
        return memory

    def attend(self, memory, mask, hidden):
        score = self.attn_v(torch.tanh(self.attn_mem(memory) + self.attn_dec(hidden).unsqueeze(1))).squeeze(-1)
        attn = F.softmax(score.masked_fill(~mask, NEG), -1)
        context = torch.bmm(attn.unsqueeze(1), memory).squeeze(1)
        return context, attn

    def step(self, prev_ids, hidden, memory, mask, copy_ids, max_oov):
        prev_base = prev_ids.masked_fill(prev_ids >= self.gen_vocab_size, 1)
        emb = self.dec_emb(prev_base)
        context, _ = self.attend(memory, mask, hidden)
        hidden = self.decoder(torch.cat([emb, context], -1), hidden)
        context, attn = self.attend(memory, mask, hidden)
        vocab = F.softmax(self.vocab_out(torch.cat([hidden, context], -1)), -1)
        p = torch.sigmoid(self.p_gen(torch.cat([hidden, context, emb], -1)))
        final = vocab.new_zeros(vocab.size(0), self.gen_vocab_size + max_oov)
        final[:, :self.gen_vocab_size] = p * vocab
        final.scatter_add_(1, copy_ids, (1 - p) * attn)
        return final, hidden, attn, p

    def forward(self, batch):
        memory = self.encode(batch); mask = batch["t_mask"]
        hidden = (memory * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        distributions = []
        for t in range(batch["decoder_in"].size(1)):
            dist, hidden, _, _ = self.step(batch["decoder_in"][:, t], hidden, memory, mask,
                                           batch["copy_ids"], batch["max_oov"])
            distributions.append(dist)
        return torch.stack(distributions, 1)

    @torch.no_grad()
    def generate(self, batch, max_len=16):
        memory = self.encode(batch); mask = batch["t_mask"]
        hidden = (memory * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        prev = torch.full((memory.size(0),), 2, dtype=torch.long, device=memory.device)
        outputs, finished = [], torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
        for _ in range(max_len):
            dist, hidden, _, _ = self.step(prev, hidden, memory, mask, batch["copy_ids"], batch["max_oov"])
            prev = dist.argmax(-1); outputs.append(prev)
            finished |= prev.eq(3)
            if finished.all(): break
        return torch.stack(outputs, 1)


def generation_loss(distributions, targets, mask):
    prob = distributions.gather(2, targets.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
    return -(prob.log() * mask).sum() / mask.sum().clamp_min(1)
