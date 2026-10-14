"""Train the from-scratch TweetQA pointer-generator on validation only."""
import json, os, random, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from .evaluate import evaluate, fmt
from .fasttext_init import fasttext_vectors
from .generative import (TweetPointerGenerator, add_generation_fields, build_gen_vocab,
                         collate_generation, generation_loss)
from .runlog import RESULTS_DIR, log_run


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def move(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def decode_ids(ids, ex, gv):
    out = []
    for i in ids:
        if i == 3: break
        if i in (0, 2): continue
        out.append(gv.itos[i] if i < len(gv) else ex["source_oovs"][i-len(gv)]
                   if i-len(gv) < len(ex["source_oovs"]) else "<unk>")
    return " ".join(out)


@torch.no_grad()
def predict(model, examples, gv, device, batch_size=64, max_len=16):
    model.eval(); out = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start+batch_size]
        batch = move(collate_generation(chunk), device)
        ids = model.generate(batch, max_len).cpu().tolist()
        out.extend(decode_ids(x, e, gv) for x, e in zip(ids, chunk))
    return out


def train(cfg, seed=1, smoke=False):
    seed_all(seed); device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows, val_rows = D.get_splits()
    if smoke: train_rows, val_rows = train_rows[:200], val_rows[:200]
    prep = dict(use_normaliser=True, max_span_len=cfg["max_span_len"], tau=cfg["tau"])
    tr, va = D.preprocess(train_rows, prep), D.preprocess(val_rows, prep)
    wv, cv = D.build_vocabs(tr, min_freq=cfg["min_freq"])
    D.encode(tr, wv, cv); D.encode(va, wv, cv)
    gv = build_gen_vocab(train_rows, cfg["gen_min_freq"])
    add_generation_fields(tr, gv, cfg["max_answer_len"])
    add_generation_fields(va, gv, cfg["max_answer_len"])
    vectors = fasttext_vectors(wv, cfg, tr, seed=seed)
    model = TweetPointerGenerator(len(wv), len(cv), len(gv), cfg, vectors).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loader = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True,
                        collate_fn=collate_generation,
                        generator=torch.Generator().manual_seed(seed))
    run_id = f"PointerGenerator-s{seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(run_id, "device", device, "params", n_params, "gen_vocab", len(gv), flush=True)
    best = None; bad = 0; history = []; best_preds = []; t0 = time.time()
    epochs = 2 if smoke else cfg["epochs"]
    for epoch in range(1, epochs+1):
        model.train(); total = tokens = 0
        for batch in loader:
            batch = move(batch, device); dist = model(batch)
            loss = generation_loss(dist, batch["target_ext"], batch["target_mask"])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]); opt.step()
            total += loss.item() * batch["target_mask"].sum().item(); tokens += batch["target_mask"].sum().item()
        preds = predict(model, va, gv, device, max_len=cfg["max_answer_len"])
        scores = evaluate(preds, [e["answers"] for e in va])
        history.append(dict(epoch=epoch, train_loss=total/max(1,tokens), **scores))
        improved = best is None or scores["rougeL"] > best["rougeL"]
        print(epoch, f"loss={total/max(1,tokens):.4f}", fmt(scores), "*" if improved else "", flush=True)
        if improved:
            best=scores; bad=0; best_preds=preds
            if not smoke:
                ckpt=Path(os.environ["A2_CKPT"]); ckpt.mkdir(parents=True,exist_ok=True)
                torch.save(dict(state_dict=model.state_dict(), cfg=cfg, seed=seed,
                                vocab=wv.itos, chars=cv.itos, gen_vocab=gv.itos), ckpt/f"{run_id}.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]: break
    if smoke:
        assert history[-1]["train_loss"] < history[0]["train_loss"]
        print("GENERATOR SMOKE PASSED"); return
    (RESULTS_DIR/"history").mkdir(parents=True,exist_ok=True)
    (RESULTS_DIR/"preds").mkdir(parents=True,exist_ok=True)
    (RESULTS_DIR/"history"/f"{run_id}.json").write_text(json.dumps(history,indent=1))
    records=[dict(row_id=e["row_id"],qid=e["qid"],pred=p,gold=e["answers"])
             for e,p in zip(va,best_preds)]
    (RESULTS_DIR/"preds"/f"{run_id}.json").write_text(json.dumps(records))
    log_run("PointerGenerator",cfg,seed,"val",best,time.time()-t0,run_id,
            f"params={n_params} device={device}:{torch.cuda.get_device_name(0) if device=='cuda' else ''}")
    print("BEST",fmt(best),run_id)


if __name__ == "__main__":
    cfg=json.loads(Path("generator_config.json").read_text())
    train(cfg, seed=1, smoke="--smoke" in os.sys.argv)
