"""B1: lexical sliding-window baseline (no learning).

For every candidate window of 1..max_len tokens that starts and ends on a
content token and contains no question content word, the score is:

  proximity = sum over tweet positions p matching a question content word
              of 1 / (1 + distance(p, window))
  + type_bonus if the window fits the expected answer type
    (who -> capitalised / @mention, when -> number / month,
     how many / how much -> number, where -> capitalised)
  - length_penalty * (window length - 1)

The two free parameters (max_len, length_penalty) are picked on the
*training* split. No learning is involved, so the validation split stays
clean for reporting.
"""
import re

try:
    from ..data import _STOP
    from ..evaluate import normalize_answer
except ImportError:
    from data import _STOP
    from evaluate import normalize_answer

_MONTHS = set("january february march april may june july august september october november december "
              "jan feb mar apr jun jul aug sep sept oct nov dec monday tuesday wednesday thursday friday "
              "saturday sunday today tonight yesterday tomorrow".split())


def _qtype(question):
    q = question.lower()
    if re.match(r"\s*(how many|how much|how old|how long)", q):
        return "num"
    first = (q.split() or [""])[0]
    return {"who": "who", "whose": "who", "who's": "who", "whom": "who",
            "when": "when", "where": "where"}.get(first, "other")


def _type_ok(qtype, toks):
    if qtype == "who" or qtype == "where":
        return all(t["raw"][:1].isupper() or t["kind"] == "mention" for t in toks)
    if qtype == "when":
        return any(t["kind"] == "number" or t["word"] in _MONTHS for t in toks)
    if qtype == "num":
        return any(t["kind"] == "number" or t["word"] in {"one", "two", "three", "four", "five", "six",
                                                          "seven", "eight", "nine", "ten"} for t in toks)
    return False


def b1_predict_one(ex, max_len=3, length_penalty=0.2, type_bonus=1.0):
    toks = ex["t_tokens"]
    qwords = {w for t in ex["q_tokens"] for w in normalize_answer(t["raw"]).split()} - _STOP
    tnorm = [set(normalize_answer(t["raw"]).split()) for t in toks]
    content = [bool(s) and not s <= _STOP for s in tnorm]
    anchors = [p for p, s in enumerate(tnorm) if s & qwords]
    qtype = _qtype(ex["question"])
    best, best_score = None, float("-inf")
    for i in range(len(toks)):
        if not content[i] or tnorm[i] & qwords:
            continue
        for j in range(i, min(len(toks), i + max_len)):
            if tnorm[j] & qwords:
                break
            if not content[j]:
                continue
            prox = sum(1.0 / (1 + (i - p if p < i else p - j)) for p in anchors)
            score = prox - length_penalty * (j - i)
            if _type_ok(qtype, toks[i:j + 1]):
                score += type_bonus
            if score > best_score:
                best, best_score = (i, j), score
    if best is None:
        return ""
    return ex["tweet"][toks[best[0]]["start"]:toks[best[1]]["end"]]


def b1_predict(examples, **kw):
    return [b1_predict_one(ex, **kw) for ex in examples]
