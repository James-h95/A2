"""B4: fine-tuned generative seq2seq (t5-small; google/flan-t5-base optional).

Input "question: <q> context: <tweet>" -> the gold answer string. Unlike the
extractive models it can produce answers that are not in the tweet
(about 35% of TweetQA). Beam search 4, max 16 new tokens.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import data as D  # noqa: E402
from src.evaluate import evaluate, fmt  # noqa: E402
from src.runlog import log_run  # noqa: E402
from src.train import get_device  # noqa: E402

B4_CFG = dict(model="t5-small", lr=3e-4, epochs=5, batch_size=16, max_src=160, max_tgt=24,
           beams=4, max_new_tokens=16, weight_decay=0.0)


def b4_src_text(r):
    return f"question: {r['question']} context: {r['tweet']}"


@torch.no_grad()
def b4_generate(model, tok, rows, device, cfg, bs=64):
    model.eval()
    out = []
    for k in range(0, len(rows), bs):
        enc = tok([b4_src_text(r) for r in rows[k:k + bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=cfg["max_src"]).to(device)
        gen = model.generate(**enc, num_beams=cfg["beams"], max_new_tokens=cfg["max_new_tokens"])
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return [o.strip() for o in out]


def b4_run(cfg, seed=1, split="val", limit=None, log=print):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = get_device()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model"]).to(device)
    train_rows, val_rows = D.get_splits()
    eval_rows = val_rows if split == "val" else D.load_test()
    if limit:
        train_rows, eval_rows = train_rows[:limit], eval_rows[:limit]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * ((len(train_rows) + cfg["batch_size"] - 1) // cfg["batch_size"])
    sched = get_linear_schedule_with_warmup(opt, 0, steps)
    rng = random.Random(seed)
    log(f"B4 {cfg['model']} device={device} train={len(train_rows)} eval={len(eval_rows)} steps={steps}")
    t0 = time.time()
    s, preds = None, None
    for ep in range(cfg["epochs"]):
        model.train()
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        tot = n = 0
        for k in range(0, len(order), cfg["batch_size"]):
            rows = [train_rows[i] for i in order[k:k + cfg["batch_size"]]]
            enc = tok([b4_src_text(r) for r in rows], return_tensors="pt", padding=True, truncation=True,
                      max_length=cfg["max_src"]).to(device)
            lab = tok([r["answers"][0] for r in rows], return_tensors="pt", padding=True, truncation=True,
                      max_length=cfg["max_tgt"]).input_ids.to(device)
            lab[lab == tok.pad_token_id] = -100
            loss = model(**enc, labels=lab).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(rows); n += len(rows)
        preds = b4_generate(model, tok, eval_rows, device, cfg)
        s = evaluate(preds, [r["answers"] for r in eval_rows])
        log(f"B4 epoch {ep+1} loss={tot/n:.4f} {split} {fmt(s)} ({(time.time()-t0)/60:.1f} min)")
    return s, preds, time.time() - t0, eval_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split", default="val")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=B4_CFG["model"])
    ap.add_argument("--epochs", type=int, default=B4_CFG["epochs"])
    a = ap.parse_args()
    cfg = {**B4_CFG, "model": a.model, "epochs": 1 if a.smoke else a.epochs}
    s, preds, tt, rows = b4_run(cfg, a.seed, a.split, limit=200 if a.smoke else None,
                             log=lambda x: print(x, flush=True))
    if not a.smoke:
        rid = log_run("B4_" + cfg["model"].split("/")[-1], cfg, a.seed, a.split, s, tt,
                      notes="fine-tuned seq2seq, beam 4 (final epoch)")
        out = ROOT / "results" / "preds"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rid}.json").write_text(json.dumps([dict(qid=r["qid"], row_id=r["row_id"], pred=p)
                                                     for r, p in zip(rows, preds)]))
    print("DONE", fmt(s), flush=True)
