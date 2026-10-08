"""TweetQA data pipeline: loading, splits, offset-preserving tokenisation,
tweet normalisation, token flags, silver-span labelling, vocabularies and
a padded DataLoader.

Design notes (see results/DATA_REPORT.md):
- test.csv is the original TweetQA dev set. It stays locked (TEST_LOCKED)
  until the final evaluation. Validation is 10% of train, grouped by tweet.
- Every token keeps its character offsets into the raw tweet, so a
  predicted span (i, j) maps back to tweet[start_i:end_j].
- The normaliser only changes the *word* used for the embedding lookup.
  The char-CNN always sees the raw (lowercased) token, so @handles stay
  distinguishable after @user -> <user>.
"""
import ast
import csv
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .evaluate import normalize_answer, rouge_l
except ImportError:  # inlined in the notebook / run as a script
    from evaluate import normalize_answer, rouge_l

TEST_LOCKED = True
DATA_DIR = Path(os.environ.get("TWEETQA_DIR", Path(__file__).resolve().parents[1] / "data"
                               if "__file__" in globals() else "data"))
SPLIT_SEED = 4012
VAL_FRAC = 0.10

# ------------------------------------------------------------------ loading

_ANSWER_ITEM = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", re.S)


def parse_answers(field):
    """The Answer column is a numpy-array repr: ['a' 'b'] (space/newline separated, mixed quotes)."""
    items = [ast.literal_eval(m) for m in _ANSWER_ITEM.findall(field)]
    if not items:
        raise ValueError(f"unparseable answer field: {field!r}")
    return [str(a) for a in items]


def _read_csv(name):
    with open(Path(DATA_DIR) / name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [dict(row_id=i, qid=r["qid"], question=r["Question"], tweet=r["Tweet"],
                 answers=parse_answers(r["Answer"]), answer_field=r["Answer"])
            for i, r in enumerate(rows)]


def _question_word(q):
    toks = tokenize(q.lower())
    w = toks[0][0] if toks else ""
    return w if w in {"what", "who", "how", "where", "why", "which", "when"} else "other"


def get_splits(seed=SPLIT_SEED, val_frac=VAL_FRAC):
    """Train/val from train.csv: exact duplicates dropped, grouped by tweet,
    stratified by the first question word of each tweet's first question."""
    rows, seen = [], set()
    for r in _read_csv("train.csv"):
        key = (r["question"], r["answer_field"], r["tweet"])
        if key not in seen:
            seen.add(key)
            rows.append(r)
    by_tweet = defaultdict(list)
    for r in rows:
        by_tweet[r["tweet"]].append(r)
    strata = defaultdict(list)
    for tweet in sorted(by_tweet):
        strata[_question_word(by_tweet[tweet][0]["question"])].append(tweet)
    rng = random.Random(seed)
    val_tweets = set()
    for s in sorted(strata):
        tweets = strata[s][:]
        rng.shuffle(tweets)
        val_tweets.update(tweets[:round(len(tweets) * val_frac)])
    train = [r for r in rows if r["tweet"] not in val_tweets]
    val = [r for r in rows if r["tweet"] in val_tweets]
    return train, val


def load_test():
    if TEST_LOCKED:
        raise RuntimeError("Test set is locked (TEST_LOCKED=True). It is used once, in Stage 8.")
    return _read_csv("test.csv")


# ------------------------------------------------------------- tokenisation

_URL = r"https?://\S+|www\.\S+"
_EMOJI_CHARS = "\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍"
_TOKEN_RE = re.compile(
    rf"(?P<url>{_URL})"
    r"|(?P<mention>@\w+)"
    r"|(?P<hashtag>#\w+)"
    rf"|(?P<emoji>[{_EMOJI_CHARS}]+)"
    r"|(?P<number>\d+(?:[.,:/]\d+)*)"
    r"|(?P<word>\w+(?:['’]\w+)*)"
    r"|(?P<sym>[^\w\s])",
    re.UNICODE,
)
_ELONG = re.compile(r"(.)\1{2,}")


def tokenize(text):
    """Returns [(raw_token, start, end, kind)] with char offsets into text."""
    return [(m.group(0), m.start(), m.end(), m.lastgroup) for m in _TOKEN_RE.finditer(text)]


def normalise_tokens(raw_tokens, use_normaliser=True):
    """Map raw tokens to model tokens: dicts with word, raw, start, end, kind.

    Normaliser: @user -> <user>, url -> <url>, emoji -> <emoji>,
    #tag -> '#' + 'tag', elongation squashed to 2 repeats, lowercase.
    With use_normaliser=False the word is just the lowercased raw token.
    """
    out = []
    for raw, s, e, kind in raw_tokens:
        if not use_normaliser:
            out.append(dict(word=raw.lower(), raw=raw, start=s, end=e, kind=kind))
        elif kind == "mention":
            out.append(dict(word="<user>", raw=raw, start=s, end=e, kind=kind))
        elif kind == "url":
            out.append(dict(word="<url>", raw=raw, start=s, end=e, kind=kind))
        elif kind == "emoji":
            out.append(dict(word="<emoji>", raw=raw, start=s, end=e, kind=kind))
        elif kind == "hashtag":
            out.append(dict(word="#", raw="#", start=s, end=s + 1, kind=kind))
            tag = raw[1:]
            out.append(dict(word=_ELONG.sub(r"\1\1", tag.lower()), raw=tag, start=s + 1, end=e, kind=kind))
        else:
            out.append(dict(word=_ELONG.sub(r"\1\1", raw.lower()), raw=raw, start=s, end=e, kind=kind))
    return out


_BYLINE_RE = re.compile(r"\(@\w+\)\s+[A-Za-z]+\.?\s+\d{1,2},\s+\d{4}\s*$")
_BOUNDARY = {".", "!", "?", "—", "–", "-", "\"", "”", "|", ":"}


def byline_start(text, tokens):
    """Index of the first token of the trailing 'Name (@handle) Month D, YYYY'
    byline, or len(tokens) if there is none. The name is taken back to the
    previous sentence boundary, at most 6 tokens."""
    m = _BYLINE_RE.search(text)
    if not m:
        return len(tokens)
    k = next((i for i, t in enumerate(tokens) if t["start"] >= m.start()), len(tokens))
    j = k
    while j > 0 and k - j < 6 and tokens[j - 1]["raw"] not in _BOUNDARY:
        j -= 1
    return j


FLAG_NAMES = ("mention", "hashtag", "url", "emoji", "number", "capitalised", "in_byline", "in_other_seq")


def token_flags(tokens, other_words, byline_from=None):
    by = len(tokens) if byline_from is None else byline_from
    flags = []
    for i, t in enumerate(tokens):
        flags.append([
            float(t["kind"] == "mention"), float(t["kind"] == "hashtag"), float(t["kind"] == "url"),
            float(t["kind"] == "emoji"), float(t["kind"] == "number"),
            float(t["raw"][:1].isupper()), float(i >= by),
            float(t["word"] in other_words and t["word"] not in _STOP),
        ])
    return flags


_STOP = set("""a an the of to in on at for by with from and or but is are was were be been being am
do does did has have had i you he she it we they me him her us them my your his its our their this that
these those what who whom whose which when where why how not no yes so if than then there here as just
about into out up down over can will would should could may might must 's ? . , ! ' " - — : ; ( )""".split())


# ------------------------------------------------------------- silver spans

def _norm_toks(raw):
    return normalize_answer(raw).split()


def silver_spans(tokens, tweet, answers, max_len=15, tau=0.5):
    """Score every span (i, j), j - i < max_len, that starts and ends on a
    content token, by token-F1 against each gold answer (max over golds).

    Returns (best, best_f1, soft):
      best  : (i, j) with the highest F1, ties broken by ROUGE-L then length
      soft  : [(i, j, w)] for all spans with F1 >= tau, weights proportional to F1
    """
    ntok = [_norm_toks(t["raw"]) for t in tokens]
    content = [bool(x) for x in ntok]
    golds = [Counter(_norm_toks(a)) for a in answers]
    glens = [sum(g.values()) for g in golds]
    scored = []
    for i in range(len(tokens)):
        if not content[i]:
            continue
        bag, n = Counter(), 0
        for j in range(i, min(len(tokens), i + max_len)):
            bag.update(ntok[j])
            n += len(ntok[j])
            if not content[j]:
                continue
            f = 0.0
            for g, gl in zip(golds, glens):
                if gl == 0:
                    continue
                ov = sum((bag & g).values())
                if ov:
                    p, r = ov / n, ov / gl
                    f = max(f, 2 * p * r / (p + r))
            if f > 0:
                scored.append((f, i, j))
    if not scored:
        return None, 0.0, []
    top = max(f for f, _, _ in scored)
    ties = [(i, j) for f, i, j in scored if f == top]
    if len(ties) > 1:
        def key(ij):
            txt = tweet[tokens[ij[0]]["start"]:tokens[ij[1]]["end"]]
            return (max(rouge_l(txt, a) for a in answers), -(ij[1] - ij[0]), -ij[0])
        ties.sort(key=key, reverse=True)
    soft = [(i, j, f) for f, i, j in scored if f >= tau]
    z = sum(w for _, _, w in soft)
    soft = [(i, j, w / z) for i, j, w in soft]
    return ties[0], top, soft


def span_text(ex, i, j):
    return ex["tweet"][ex["t_tokens"][i]["start"]:ex["t_tokens"][j]["end"]]


# ------------------------------------------------------------------- vocab

class Vocab:
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, counter, min_freq=2, specials=()):
        self.itos = [self.PAD, self.UNK] + [s for s in specials if s not in (self.PAD, self.UNK)]
        self.itos += sorted([w for w, c in counter.items() if c >= min_freq and w not in self.itos],
                            key=lambda w: (-counter[w], w))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def __call__(self, w):
        return self.stoi.get(w, 1)


SPECIALS = ("<user>", "<url>", "<emoji>")
MAX_WORD_CHARS = 16


def build_vocabs(examples, min_freq=2, char_min_freq=5):
    """Vocabularies come from the training split only."""
    wc, cc = Counter(), Counter()
    for ex in examples:
        for t in ex["t_tokens"] + ex["q_tokens"]:
            wc[t["word"]] += 1
            cc.update(t["raw"].lower()[:MAX_WORD_CHARS])
    return Vocab(wc, min_freq, SPECIALS), Vocab(cc, char_min_freq)


# ----------------------------------------------------------------- encoding

DEFAULT_PREP = dict(use_normaliser=True, max_span_len=15, tau=0.5)


def preprocess(rows, prep=None):
    """Tokenise, normalise, flag and silver-label raw rows."""
    prep = {**DEFAULT_PREP, **(prep or {})}
    out = []
    for r in rows:
        t_tokens = normalise_tokens(tokenize(r["tweet"]), prep["use_normaliser"])
        q_tokens = normalise_tokens(tokenize(r["question"]), prep["use_normaliser"])
        by = byline_start(r["tweet"], t_tokens)
        t_words = {t["word"] for t in t_tokens}
        q_words = {t["word"] for t in q_tokens}
        best, best_f1, soft = silver_spans(t_tokens, r["tweet"], r["answers"], prep["max_span_len"], prep["tau"])
        out.append(dict(
            r, t_tokens=t_tokens, q_tokens=q_tokens, byline_from=by,
            t_flags=token_flags(t_tokens, q_words, by), q_flags=token_flags(q_tokens, t_words),
            best=best, best_f1=best_f1, soft=soft,
            trainable=best is not None and best_f1 >= prep["tau"],
        ))
    return out


def encode(examples, wvocab, cvocab):
    for ex in examples:
        for side in ("t", "q"):
            toks = ex[f"{side}_tokens"]
            ex[f"{side}_ids"] = [wvocab(t["word"]) for t in toks]
            ex[f"{side}_chars"] = [[cvocab(c) for c in t["raw"].lower()[:MAX_WORD_CHARS]] or [1] for t in toks]
    return examples


class QADataset(Dataset):
    def __init__(self, examples, train=False):
        # during training, only examples with a usable silver span contribute to the loss
        self.examples = [e for e in examples if e["trainable"]] if train else list(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def _pad_side(batch, side, n_flags):
    B = len(batch)
    T = max(1, max(len(ex[f"{side}_ids"]) for ex in batch))
    W = max(1, max((len(c) for ex in batch for c in ex[f"{side}_chars"]), default=1))
    words = torch.zeros(B, T, dtype=torch.long)
    chars = torch.zeros(B, T, W, dtype=torch.long)
    flags = torch.zeros(B, T, n_flags)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b, ex in enumerate(batch):
        n = len(ex[f"{side}_ids"])
        words[b, :n] = torch.tensor(ex[f"{side}_ids"], dtype=torch.long)
        for k, cs in enumerate(ex[f"{side}_chars"]):
            chars[b, k, :len(cs)] = torch.tensor(cs, dtype=torch.long)
        if n:
            flags[b, :n] = torch.tensor(ex[f"{side}_flags"])
        mask[b, :n] = True
    return words, chars, flags, mask


def collate(batch):
    tw, tc, tf, tm = _pad_side(batch, "t", len(FLAG_NAMES))
    qw, qc, qf, qm = _pad_side(batch, "q", len(FLAG_NAMES))
    B, T = tw.shape
    y_start = torch.full((B,), -1, dtype=torch.long)
    y_end = torch.full((B,), -1, dtype=torch.long)
    soft_start = torch.zeros(B, T)
    soft_end = torch.zeros(B, T)
    for b, ex in enumerate(batch):
        if ex["best"] is not None:
            y_start[b], y_end[b] = ex["best"]
        for i, j, w in ex["soft"]:
            soft_start[b, i] += w
            soft_end[b, j] += w
    return dict(t_words=tw, t_chars=tc, t_flags=tf, t_mask=tm,
                q_words=qw, q_chars=qc, q_flags=qf, q_mask=qm,
                y_start=y_start, y_end=y_end, soft_start=soft_start, soft_end=soft_end,
                meta=batch)


def make_loader(examples, batch_size=32, train=False, seed=0):
    ds = QADataset(examples, train=train)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, collate_fn=collate, generator=g)


# ------------------------------------------------------------------ oracle

def oracle_predictions(examples):
    """Best silver span text for every example ('' if there is no overlapping span)."""
    return [span_text(ex, *ex["best"]) if ex["best"] is not None else "" for ex in examples]
