import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D  # noqa: E402
from src.evaluate import (evaluate, lcs_len, normalize_answer, rouge_l,  # noqa: E402
                          score_example, token_f1)

TWEET = ("Our prayers are with the students at Independence High School & all the first "
         "responders on the scene. #PatriotPride— Doug Ducey (@dougducey) February 12, 2016")


# ---------------------------------------------------------------- evaluate

def test_gold_vs_gold_is_100():
    golds = [["independence high school"], ["w nj", "w nj"], ["5:09 pm"]]
    s = evaluate([g[0] for g in golds], golds)
    for m, v in s.items():
        if m != "meteor":
            assert v == pytest.approx(100.0), m
    # METEOR's fragmentation penalty is 0.5*(1/m)^3 even for a perfect match of m tokens,
    # so identical answers score 1 - 0.5/m^3 (3, 2 and 3 normalised tokens here)
    expected = 100 * sum(1 - 0.5 / m ** 3 for m in (3, 2, 3)) / 3
    assert s["meteor"] == pytest.approx(expected)


def test_empty_prediction_is_zero():
    s = evaluate(["", ""], [["bob simon"], ["mike"]])
    assert all(v == 0.0 for v in s.values())


def test_max_over_golds():
    s = score_example("helping economy", ["completely different", "helping economy"])
    assert s["em"] == 1.0 and s["f1"] == 1.0


def test_normalise_unicode_punct_and_articles():
    assert normalize_answer("The #ThisEndsNow.—") == "thisendsnow"
    assert normalize_answer("“Hello”, World!") == "hello world"


def test_rouge_l_known_value():
    assert lcs_len("a b c d".split(), "a c d e".split()) == 3
    # pred 2 tokens, gold 3 tokens, LCS 2: P=1, R=2/3, beta=1.2
    p, r, b = 1.0, 2 / 3, 1.2
    assert rouge_l("high school", "independence high school") == pytest.approx((1 + b * b) * p * r / (r + b * b * p))


def test_token_f1_partial():
    assert token_f1("high school", "independence high school") == pytest.approx(0.8)


# ------------------------------------------------------------------ data

def test_parse_answers_formats():
    assert D.parse_answers("['w nj' 'w nj']") == ["w nj", "w nj"]
    assert D.parse_answers("[\"he's here\" 'x']") == ["he's here", "x"]
    assert D.parse_answers("['a b'\n 'c']") == ["a b", "c"]


def test_offsets_roundtrip():
    for tok, s, e, _ in D.tokenize(TWEET):
        assert TWEET[s:e] == tok
    toks = D.normalise_tokens(D.tokenize(TWEET))
    for t in toks:
        assert TWEET[t["start"]:t["end"]] == t["raw"]


def test_normaliser_rules():
    toks = D.normalise_tokens(D.tokenize("Sooooo good @Bob #GoTeam 😂😂 http://x.co/a"))
    words = [t["word"] for t in toks]
    assert words == ["soo", "good", "<user>", "#", "goteam", "<emoji>", "<url>"]
    raw_words = [t["word"] for t in D.normalise_tokens(D.tokenize("Sooooo @Bob #GoTeam"), use_normaliser=False)]
    assert raw_words == ["sooooo", "@bob", "#goteam"]


def test_byline_detection():
    toks = D.normalise_tokens(D.tokenize(TWEET))
    k = D.byline_start(TWEET, toks)
    assert [t["raw"] for t in toks[k:k + 2]] == ["Doug", "Ducey"]


def test_silver_span_exact_substring():
    toks = D.normalise_tokens(D.tokenize(TWEET))
    best, f1, soft = D.silver_spans(toks, TWEET, ["independence high school"])
    assert f1 == 1.0
    assert TWEET[toks[best[0]]["start"]:toks[best[1]]["end"]] == "Independence High School"
    assert abs(sum(w for *_, w in soft) - 1) < 1e-9
    assert all(w > 0 for *_, w in soft)


def test_silver_span_hashtag_and_byline():
    toks = D.normalise_tokens(D.tokenize(TWEET))
    best, f1, _ = D.silver_spans(toks, TWEET, ["#patriotpride"])
    assert f1 == 1.0 and TWEET[toks[best[0]]["start"]:toks[best[1]]["end"]] == "PatriotPride"
    best, f1, _ = D.silver_spans(toks, TWEET, ["doug ducey"])
    assert f1 == 1.0 and TWEET[toks[best[0]]["start"]:toks[best[1]]["end"]] == "Doug Ducey"


def test_silver_span_none_when_no_overlap():
    toks = D.normalise_tokens(D.tokenize(TWEET))
    best, f1, soft = D.silver_spans(toks, TWEET, ["completely unrelated"])
    assert best is None and f1 == 0 and soft == []


def test_test_set_locked():
    assert D.TEST_LOCKED
    with pytest.raises(RuntimeError):
        D.load_test()


def test_splits_grouped_and_deterministic():
    tr, va = D.get_splits()
    assert not ({r["tweet"] for r in tr} & {r["tweet"] for r in va})
    tr2, va2 = D.get_splits()
    assert [r["row_id"] for r in va] == [r["row_id"] for r in va2]
    assert 0.08 < len(va) / (len(tr) + len(va)) < 0.12


def test_collate_shapes():
    rows = [dict(row_id=0, qid="a", question="who leads the school?", tweet=TWEET,
                 answers=["doug ducey"], answer_field="")] * 2
    exs = D.preprocess(rows)
    wv, cv = D.build_vocabs(exs, min_freq=1, char_min_freq=1)
    D.encode(exs, wv, cv)
    b = D.collate(exs)
    B, T = b["t_words"].shape
    assert b["t_chars"].shape[:2] == (B, T) and b["t_flags"].shape == (B, T, len(D.FLAG_NAMES))
    assert b["soft_start"].sum(1).allclose(b["soft_start"].new_ones(B))


# ------------------------------------------------------------------ model

def test_bilstm_matches_packed_reference():
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from src.model import BiLSTM
    torch.manual_seed(0)
    m = BiLSTM(5, 4, layers=2).eval()
    ref = torch.nn.LSTM(5, 4, num_layers=2, batch_first=True, bidirectional=True).eval()
    # copy our per-direction weights into the reference cuDNN-style LSTM
    for l in range(2):
        for d, src in (("", m.fw[l]), ("_reverse", m.bw[l])):
            for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                getattr(ref, f"{n}_l{l}{d}").data.copy_(getattr(src, f"{n}_l0"))
    x = torch.randn(3, 7, 5)
    lengths = torch.tensor([7, 4, 1])
    mask = torch.arange(7).unsqueeze(0) < lengths.unsqueeze(1)
    ours = m(x, mask)
    packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    theirs = pad_packed_sequence(ref(packed)[0], batch_first=True, total_length=7)[0]
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_decode_respects_band():
    import torch
    from src.model import decode
    s = torch.tensor([[0.0, 5.0, 0.0, 0.0]])
    e = torch.tensor([[9.0, 0.0, 0.0, 5.0]])     # best end (0) is before best start (1): not allowed
    idx, _ = decode(s, e, max_len=3)
    assert idx.tolist() == [[1, 3]]
    idx, _ = decode(s, e, max_len=2)
    i, j = idx[0].tolist()
    assert i <= j < i + 2
