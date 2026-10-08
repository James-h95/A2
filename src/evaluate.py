"""Evaluation metrics for TweetQA.

Every metric is computed per example as the max over the gold answers,
and corpus scores are the mean x 100.

- BLEU-1: nltk sentence_bleu, weights (1,0,0,0), smoothing method1
- METEOR: nltk meteor_score (needs WordNet)
- ROUGE-L: our own LCS implementation, F-measure with beta = 1.2 (the
  coco-caption convention used by the official TweetQA evaluation)
- EM and token-F1: SQuAD-style, computed on normalised text
"""
import re
import string
import unicodedata
from collections import Counter

METRICS = ("bleu1", "meteor", "rougeL", "em", "f1")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_nltk_ready = False


def _ensure_nltk():
    global _nltk_ready
    if _nltk_ready:
        return
    import nltk
    for pkg, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
    _nltk_ready = True


def normalize_answer(s):
    """Lowercase, replace ASCII and Unicode punctuation with spaces,
    drop articles, collapse whitespace."""
    s = s.lower()
    s = "".join(" " if (ch in string.punctuation or unicodedata.category(ch).startswith("P")) else ch
                for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def norm_tokens(s):
    return normalize_answer(s).split()


# ---------------------------------------------------------------- metrics

def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred, gold):
    p, g = norm_tokens(pred), norm_tokens(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def lcs_len(a, b):
    """Length of the longest common subsequence of two token lists (O(len(a)*len(b)) DP)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred, gold, beta=1.2):
    p, g = norm_tokens(pred), norm_tokens(gold)
    if not p or not g:
        return float(p == g and bool(p))
    lcs = lcs_len(p, g)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p), lcs / len(g)
    return (1 + beta ** 2) * prec * rec / (rec + beta ** 2 * prec)


def bleu1(pred, gold):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    p, g = norm_tokens(pred), norm_tokens(gold)
    if not p or not g:
        return 0.0
    return sentence_bleu([g], p, weights=(1, 0, 0, 0),
                         smoothing_function=SmoothingFunction().method1)


def meteor(pred, gold):
    _ensure_nltk()
    from nltk.translate.meteor_score import meteor_score
    p, g = norm_tokens(pred), norm_tokens(gold)
    if not p or not g:
        return 0.0
    return meteor_score([g], p)


_FUNCS = {"bleu1": bleu1, "meteor": meteor, "rougeL": rouge_l, "em": exact_match, "f1": token_f1}


def score_example(pred, golds, metrics=METRICS):
    """Per-example scores in [0, 1]: max over gold answers."""
    pred = pred or ""
    return {m: max(_FUNCS[m](pred, g) for g in golds) for m in metrics}


def evaluate(preds, golds_list, metrics=METRICS, return_per_example=False):
    """preds: list[str]; golds_list: list[list[str]]. Returns corpus scores (mean x 100)."""
    assert len(preds) == len(golds_list), (len(preds), len(golds_list))
    per = [score_example(p, g, metrics) for p, g in zip(preds, golds_list)]
    n = max(1, len(per))
    corpus = {m: 100.0 * sum(s[m] for s in per) / n for m in metrics}
    return (corpus, per) if return_per_example else corpus


def fmt(scores):
    return " ".join(f"{m}={scores[m]:.2f}" for m in METRICS if m in scores)
