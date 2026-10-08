"""B5: open instruct LLM (Qwen/Qwen2.5-1.5B-Instruct) with a fixed few-shot prompt,
greedy decoding, output stripped to a short answer. No fine-tuning.

The prompt is recorded verbatim in PROMPT_SYSTEM / the few-shot block below. The
few-shot examples are fixed training-split rows chosen by row id.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import data as D  # noqa: E402
from src.evaluate import evaluate, fmt  # noqa: E402
from src.runlog import log_run  # noqa: E402
from src.train import get_device  # noqa: E402

B5_CFG = dict(model="Qwen/Qwen2.5-1.5B-Instruct", shots=4, max_new_tokens=16, batch_size=16)
PROMPT_SYSTEM = ("You answer questions about a tweet. Reply with only the short answer "
                 "(usually a few words copied from the tweet), with no explanation.")
SHOT_QWORDS = ("who", "what", "when", "how")   # one fixed train example per question type


def pick_shots(train_rows, k):
    shots = []
    for w in SHOT_QWORDS[:k]:
        for r in train_rows:        # first train row (fixed order) with this q-word and a short extractive answer
            a = r["answers"][0]
            if r["question"].lower().startswith(w) and a.lower() in r["tweet"].lower() and len(a.split()) <= 4:
                shots.append(r)
                break
    return shots


def b5_messages(r, shots):
    msgs = [dict(role="system", content=PROMPT_SYSTEM)]
    for s in shots:
        msgs.append(dict(role="user", content=f"Tweet: {s['tweet']}\nQuestion: {s['question']}"))
        msgs.append(dict(role="assistant", content=s["answers"][0]))
    msgs.append(dict(role="user", content=f"Tweet: {r['tweet']}\nQuestion: {r['question']}"))
    return msgs


def b5_clean(text):
    text = text.strip().split("\n")[0]
    text = re.sub(r"^(answer|a)\s*:\s*", "", text, flags=re.I)
    return text.strip().strip('"').rstrip(".")


def b5_run(cfg, split="val", limit=None, log=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    device = get_device()
    tok = AutoTokenizer.from_pretrained(cfg["model"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float16
                                                 if device != "cpu" else torch.float32).to(device).eval()
    train_rows, val_rows = D.get_splits()
    rows = val_rows if split == "val" else D.load_test()
    if limit:
        rows = rows[:limit]
    shots = pick_shots(train_rows, cfg["shots"])
    log(f"B5 {cfg['model']} device={device} eval={len(rows)} shots={[s['row_id'] for s in shots]}")
    log("PROMPT (verbatim, last user turn is the target):\n" + json.dumps(b5_messages(rows[0], shots), indent=1))
    t0 = time.time()
    preds = []
    for k in range(0, len(rows), cfg["batch_size"]):
        chunk = rows[k:k + cfg["batch_size"]]
        texts = [tok.apply_chat_template(b5_messages(r, shots), tokenize=False, add_generation_prompt=True)
                 for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=cfg["max_new_tokens"],
                                 pad_token_id=tok.eos_token_id)
        preds += [b5_clean(t) for t in tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
        if k % (cfg["batch_size"] * 10) == 0:
            log(f"  {k + len(chunk)}/{len(rows)} ({(time.time()-t0)/60:.1f} min)")
    s = evaluate(preds, [r["answers"] for r in rows])
    return s, preds, time.time() - t0, rows, shots


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=B5_CFG["model"])
    ap.add_argument("--shots", type=int, default=B5_CFG["shots"])
    a = ap.parse_args()
    cfg = {**B5_CFG, "model": a.model, "shots": a.shots}
    s, preds, tt, rows, shots = b5_run(cfg, a.split, limit=32 if a.smoke else None, log=lambda x: print(x, flush=True))
    for r, p in list(zip(rows, preds))[:8]:
        print(f"  Q: {r['question']} | gold: {r['answers']} | pred: {p}")
    if not a.smoke:
        cfg = {**cfg, "prompt_system": PROMPT_SYSTEM, "shot_row_ids": [x["row_id"] for x in shots]}
        rid = log_run("B5_" + cfg["model"].split("/")[-1] + f"_{cfg['shots']}shot", cfg, 0, a.split, s, tt,
                      notes="few-shot instruct LLM, greedy, no fine-tuning")
        out = ROOT / "results" / "preds"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rid}.json").write_text(json.dumps([dict(qid=r["qid"], row_id=r["row_id"], pred=p)
                                                     for r, p in zip(rows, preds)]))
    print("DONE", fmt(s), flush=True)
