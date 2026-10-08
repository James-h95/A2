"""Training loop for TweetBiDAF: config dict -> train / early stop / checkpoint / runs.csv.

Usage (from A2/):
  python -m src.train --name main --seed 1                        # full run
  python -m src.train --name smoke --smoke                        # 200 examples, 2 epochs
  python -m src.train --name a3 --set use_self_attn=false --seed 2
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from . import data as D
    from .evaluate import evaluate, fmt
    from .model import TweetBiDAF, decode, span_loss
    from .runlog import RESULTS_DIR, log_run
except ImportError:
    import data as D
    from evaluate import evaluate, fmt
    from model import TweetBiDAF, decode, span_loss
    from runlog import RESULTS_DIR, log_run

DEFAULT_TRAIN = dict(
    lr=1e-3, batch_size=32, epochs=30, patience=4, grad_clip=5.0, weight_decay=0.0,
    use_normaliser=True, tau=0.5, soft_tau=None, soft_spans=True, min_freq=2, word_init="random",
    max_span_len=15,
)
CKPT_DIR = Path(os.environ.get("A2_CKPT", RESULTS_DIR.parent / "checkpoints"))


def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def prepare_data(cfg, train_rows=None, val_rows=None, limit=None):
    """Preprocess and encode the train/val splits. Vocab comes from the training split only."""
    if train_rows is None:
        train_rows, val_rows = D.get_splits()
    if limit:
        train_rows, val_rows = train_rows[:limit], val_rows[:limit]
    prep = dict(use_normaliser=cfg["use_normaliser"], max_span_len=cfg["max_span_len"], tau=cfg["tau"],
                soft_tau=cfg.get("soft_tau"))
    tr, va = D.preprocess(train_rows, prep), D.preprocess(val_rows, prep)
    wv, cv = D.build_vocabs(tr, min_freq=cfg["min_freq"])
    D.encode(tr, wv, cv)
    D.encode(va, wv, cv)
    return tr, va, wv, cv


@torch.no_grad()
def predict(model, examples, device, batch_size=64, max_span_len=15):
    model.eval()
    preds, spans = [], []
    for batch in D.make_loader(examples, batch_size=batch_size):
        meta = batch["meta"]
        out = model(to_device(batch, device))
        idx, _ = decode(out["start_logits"], out["end_logits"], max_span_len)
        for ex, (i, j) in zip(meta, idx.tolist()):
            spans.append((i, j))
            preds.append(D.span_text(ex, i, j))
    return preds, spans


def build_model(cfg, wv, cv, word_vectors=None):
    mcfg = {k: cfg[k] for k in _default_model() if k in cfg}
    return TweetBiDAF(len(wv), len(cv), mcfg, word_vectors)


def train(cfg=None, seed=1, name="main", limit=None, data=None, log=print, save=True):
    """Train one model. Returns dict(best scores, history, checkpoint path, run_id)."""
    cfg = {**DEFAULT_TRAIN, **_default_model(), **(cfg or {})}
    seed_everything(seed)
    device = get_device()
    tr, va, wv, cv = data if data is not None else prepare_data(cfg, limit=limit)
    wvecs = None
    if cfg["word_init"] == "fasttext":
        try:
            from .fasttext_init import fasttext_vectors
        except ImportError:
            from fasttext_init import fasttext_vectors
        wvecs = fasttext_vectors(wv, cfg, tr, seed=seed)
    model = build_model(cfg, wv, cv, wvecs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = D.make_loader(tr, batch_size=cfg["batch_size"], train=True, seed=seed)
    golds = [e["answers"] for e in va]
    run_id = f"{name}-s{seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    log(f"[{run_id}] device={device} params={n_params:,} train_used={len(loader.dataset)}/{len(tr)} "
        f"val={len(va)} vocab={len(wv)} chars={len(cv)}")
    log(f"[{run_id}] config={json.dumps(cfg, sort_keys=True)}")

    best, best_epoch, bad, history, best_preds = None, -1, 0, [], []
    ckpt = CKPT_DIR / f"{run_id}.pt"
    t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for batch in loader:
            batch = to_device(batch, device)
            loss = span_loss(model(batch), batch, soft=cfg["soft_spans"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tot += loss.item() * batch["t_words"].size(0)
            n += batch["t_words"].size(0)
        train_time = time.time() - t0
        preds, spans = predict(model, va, device, max_span_len=cfg["max_span_len"])
        scores = evaluate(preds, golds)
        history.append(dict(epoch=epoch, train_loss=tot / max(1, n), epoch_time=train_time, **scores))
        improved = best is None or scores["rougeL"] > best["rougeL"]
        log(f"[{run_id}] epoch {epoch:2d} loss={tot / max(1, n):.4f} time={train_time:.1f}s val {fmt(scores)}"
            + (" *" if improved else ""))
        if improved:
            best, best_epoch, bad = scores, epoch, 0
            best_preds = [dict(row_id=e["row_id"], qid=e["qid"], pred=p, span=sp)
                          for e, p, sp in zip(va, preds, spans)]
            if save:
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(dict(state_dict=model.state_dict(), cfg=cfg, seed=seed, epoch=epoch,
                                vocab=wv.itos, chars=cv.itos), ckpt)
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"[{run_id}] early stop at epoch {epoch} (best epoch {best_epoch})")
                break
    total = time.time() - t_start
    log(f"[{run_id}] best epoch {best_epoch}: val {fmt(best)} | total {total/60:.1f} min")
    if save:   # smoke tests (save=False) leave no files behind, so results/ only holds logged runs
        hist_dir = RESULTS_DIR / "history"
        hist_dir.mkdir(parents=True, exist_ok=True)
        (hist_dir / f"{run_id}.json").write_text(json.dumps(dict(cfg=cfg, seed=seed, history=history,
                                                                 best_epoch=best_epoch), indent=1))
        pred_dir = RESULTS_DIR / "preds"
        pred_dir.mkdir(parents=True, exist_ok=True)
        (pred_dir / f"{run_id}.json").write_text(json.dumps(best_preds))
        log_run(name, cfg, seed, "val", best, total, run_id=run_id,
                notes=f"best_epoch={best_epoch} params={n_params} limit={limit} device={device}"
                + (f":{torch.cuda.get_device_name(0)}" if device == "cuda" else ""))
    return dict(best=best, best_epoch=best_epoch, history=history, ckpt=str(ckpt), run_id=run_id,
                model=model, data=(tr, va, wv, cv))


def _default_model():
    try:
        from .model import DEFAULT_MODEL
    except ImportError:
        from model import DEFAULT_MODEL
    return dict(DEFAULT_MODEL)


def load_checkpoint(path, device=None):
    device = device or get_device()
    ck = torch.load(path, map_location="cpu", weights_only=False)
    wv, cv = D.Vocab({}, 1), D.Vocab({}, 1)
    wv.itos, cv.itos = ck["vocab"], ck["chars"]
    wv.stoi = {w: i for i, w in enumerate(wv.itos)}
    cv.stoi = {c: i for i, c in enumerate(cv.itos)}
    model = build_model(ck["cfg"], wv, cv)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["cfg"], wv, cv


def _parse_set(items):
    """Parse key=json_value overrides. Unknown keys and values whose type doesn't match the default
    (e.g. a bool flag given a string) are errors, so a typo can never silently change an experiment."""
    defaults = {**DEFAULT_TRAIN, **_default_model()}
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in defaults:
            raise SystemExit(f"--set: unknown config key {k!r}")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        d = defaults[k]
        ok = (d is None or val is None or isinstance(val, type(d))
              or (isinstance(d, float) and isinstance(val, int) and not isinstance(val, bool))
              or (isinstance(d, tuple) and isinstance(val, list)))
        if isinstance(d, bool) and not isinstance(val, bool):
            ok = False
        if not ok:
            raise SystemExit(f"--set: {k}={v!r} has type {type(val).__name__}, expected {type(d).__name__}")
        out[k] = val
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="200 examples, 2 epochs, no runs.csv entry")
    ap.add_argument("--set", nargs="*", help="config overrides key=json_value")
    args = ap.parse_args()
    cfg = _parse_set(args.set)
    if args.smoke:
        cfg.update(epochs=2, patience=99)
    res = train(cfg, seed=args.seed, name=args.name, limit=200 if args.smoke else None,
                log=lambda s: print(s, flush=True), save=not args.smoke)
    if args.smoke:
        h = res["history"]
        assert h[-1]["train_loss"] < h[0]["train_loss"], "smoke test: loss did not decrease"
        tr, va, wv, cv = res["data"]
        model = res["model"]
        batch = to_device(D.collate(va[:4]), get_device())
        with torch.no_grad():
            model.eval()
            o = model(batch, return_attn=True)
        B, T = batch["t_words"].shape
        J = batch["q_words"].shape[1]
        assert o["S"].shape == (B, T, J) and o["c2q"].shape == (B, T, J)
        if o["q2c"] is not None:
            assert o["q2c"].shape == (B, T)
        if o["self_attn"] is not None:
            assert o["self_attn"].shape == (B, T, T)
        assert torch.allclose(o["c2q"].sum(-1), torch.ones(B, T, device=o["c2q"].device), atol=1e-4)
        print(f"SMOKE OK: loss {h[0]['train_loss']:.3f} -> {h[-1]['train_loss']:.3f}; "
              f"attn shapes S{tuple(o['S'].shape)} self{tuple(o['self_attn'].shape) if o['self_attn'] is not None else None}")
    sys.exit(0)
