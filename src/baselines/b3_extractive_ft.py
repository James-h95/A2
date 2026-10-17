"""B3: fine-tuned pretrained encoder (vinai/bertweet-base, fallback roberta-base)
with a span head, trained on the same silver spans as the main model.

Each of our word tokens is BPE-encoded separately. That keeps a subword -> word
map, so the predicted subword span maps back to our tokens and then to raw
tweet text through the same char offsets the main model uses. This works with
BERTweet's slow tokenizer, which has no offset mapping.

Input: <s> question </s></s> tweet </s>, truncated to max_len subwords
(the tweet is truncated first; this is rare at 128).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import data as D  # noqa: E402
from src.evaluate import evaluate, fmt  # noqa: E402
from src.runlog import log_run  # noqa: E402
from src.train import get_device  # noqa: E402

B3_CFG = dict(model="vinai/bertweet-base", lr=3e-5, epochs=3, batch_size=16, max_len=128,
           warmup=0.1, weight_decay=0.01, max_answer_subwords=30)


def b3_word_piece(tok, t):
    raw = t["raw"]
    if t["kind"] == "mention":
        raw = "@USER" if "bertweet" in tok.name_or_path else raw
    elif t["kind"] == "url":
        raw = "HTTPURL" if "bertweet" in tok.name_or_path else raw
    ids = tok.encode(" " + raw if "roberta" in tok.name_or_path else raw, add_special_tokens=False)
    return ids or [tok.unk_token_id]


def b3_featurise(tok, ex, max_len):
    q_ids = [i for t in ex["q_tokens"] for i in b3_word_piece(tok, t)][:40]
    ids = [tok.cls_token_id] + q_ids + [tok.sep_token_id, tok.sep_token_id]
    word_of = [-1] * len(ids)
    budget = max_len - len(ids) - 1
    for w, t in enumerate(ex["t_tokens"]):
        p = b3_word_piece(tok, t)
        if len(p) > budget:
            break
        ids += p
        word_of += [w] * len(p)
        budget -= len(p)
    ids.append(tok.sep_token_id)
    word_of.append(-1)
    ys = ye = -1
    if ex["best"] is not None:
        i, j = ex["best"]
        if i in word_of and j in word_of:
            ys = word_of.index(i)
            ye = len(word_of) - 1 - word_of[::-1].index(j)
    return dict(ids=ids, word_of=word_of, ys=ys, ye=ye)


def b3_batches(feats, bs, shuffle, rng, pad_id):
    order = list(range(len(feats)))
    if shuffle:
        rng.shuffle(order)
    for k in range(0, len(order), bs):
        fs = [feats[i] for i in order[k:k + bs]]
        T = max(len(f["ids"]) for f in fs)
        ids = torch.full((len(fs), T), pad_id, dtype=torch.long)
        att = torch.zeros(len(fs), T, dtype=torch.long)
        ctx = torch.zeros(len(fs), T, dtype=torch.bool)
        for b, f in enumerate(fs):
            ids[b, :len(f["ids"])] = torch.tensor(f["ids"])
            att[b, :len(f["ids"])] = 1
            ctx[b, :len(f["ids"])] = torch.tensor([w >= 0 for w in f["word_of"]])
        yield order[k:k + bs], ids, att, ctx, torch.tensor([f["ys"] for f in fs]), torch.tensor([f["ye"] for f in fs])


class SpanModel(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 2)

    def forward(self, ids, att, ctx):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        s, e = self.head(h).unbind(-1)
        return s.masked_fill(~ctx, -1e4), e.masked_fill(~ctx, -1e4)


@torch.no_grad()
def b3_predict(model, exs, feats, device, cfg, pad_id):
    model.eval()
    preds = [""] * len(exs)
    for idx, ids, att, ctx, _, _ in b3_batches(feats, 64, False, None, pad_id):
        s, e = model(ids.to(device), att.to(device), ctx.to(device))
        ls, le = F.log_softmax(s, -1), F.log_softmax(e, -1)
        T = ls.size(1)
        sc = ls.unsqueeze(2) + le.unsqueeze(1)
        r = torch.arange(T, device=device)
        band = (r[None] >= r[:, None]) & (r[None] - r[:, None] < cfg["max_answer_subwords"])
        flat = sc.masked_fill(~band, -1e9).view(len(idx), -1).argmax(-1).tolist()
        for k, f in zip(idx, flat):
            i, j = feats[k]["word_of"][f // T], feats[k]["word_of"][f % T]
            if i >= 0 and j >= i:
                preds[k] = D.span_text(exs[k], i, j)
    return preds


def b3_run(cfg, seed=1, split="val", limit=None, log=print):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = get_device()
    try:
        tok = AutoTokenizer.from_pretrained(cfg["model"], use_fast=False)
    except Exception as e:  # noqa: BLE001
        log(f"could not load {cfg['model']} ({e}); falling back to roberta-base")
        cfg = {**cfg, "model": "roberta-base"}
        tok = AutoTokenizer.from_pretrained(cfg["model"])
    train_rows, val_rows = D.get_splits()
    eval_rows = val_rows if split == "val" else D.load_test()
    if limit:
        train_rows, eval_rows = train_rows[:limit], eval_rows[:limit]
    tr, ev = D.preprocess(train_rows), D.preprocess(eval_rows)
    ftr = [b3_featurise(tok, e, cfg["max_len"]) for e in tr if e["trainable"]]
    ftr = [f for f in ftr if f["ys"] >= 0]
    fev = [b3_featurise(tok, e, cfg["max_len"]) for e in ev]
    model = SpanModel(cfg["model"]).to(device)
    nd = ["bias", "LayerNorm.weight", "norm"]
    groups = [dict(params=[p for n, p in model.named_parameters() if not any(x in n for x in nd)],
                   weight_decay=cfg["weight_decay"]),
              dict(params=[p for n, p in model.named_parameters() if any(x in n for x in nd)], weight_decay=0.0)]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    steps = cfg["epochs"] * ((len(ftr) + cfg["batch_size"] - 1) // cfg["batch_size"])
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * steps), steps)
    rng = random.Random(seed)
    log(f"B3 {cfg['model']} device={device} train_feats={len(ftr)} eval={len(fev)} steps={steps}")
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        tot = n = 0
        for _, ids, att, ctx, ys, ye in b3_batches(ftr, cfg["batch_size"], True, rng, tok.pad_token_id):
            ids, att, ctx, ys, ye = (x.to(device) for x in (ids, att, ctx, ys, ye))
            s, e = model(ids, att, ctx)
            loss = F.cross_entropy(s, ys) + F.cross_entropy(e, ye)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(ys); n += len(ys)
        preds = b3_predict(model, ev, fev, device, cfg, tok.pad_token_id)
        s = evaluate(preds, [e["answers"] for e in ev])
        log(f"B3 epoch {ep+1} loss={tot/n:.4f} {split} {fmt(s)} ({(time.time()-t0)/60:.1f} min)")
    return s, preds, time.time() - t0, cfg, ev


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", default="val")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=B3_CFG["model"])
    a = ap.parse_args()
    if a.split == "test":
        # Final test evaluation (Stage 8 only): unlock explicitly so it is visible in the log.
        D.TEST_LOCKED = False
        print("TEST SET UNLOCKED for final evaluation (fixed config, no tuning on test)", flush=True)
    cfg = {**B3_CFG, "model": a.model}
    if a.smoke:
        cfg["epochs"] = 1
    s, preds, tt, cfg, ev = b3_run(cfg, a.seed, a.split, limit=200 if a.smoke else None,
                                log=lambda x: print(x, flush=True))
    if not a.smoke:
        rid = log_run("B3_" + cfg["model"].split("/")[-1], cfg, a.seed, a.split, s, tt,
                      notes="fine-tuned pretrained encoder + span head on silver spans (final epoch)")
        out = ROOT / "results" / "preds"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rid}.json").write_text(json.dumps([dict(qid=e["qid"], row_id=e["row_id"], pred=p)
                                                     for e, p in zip(ev, preds)]))
    print("DONE", fmt(s), flush=True)
