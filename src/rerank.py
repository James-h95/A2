"""Deterministic lightweight reranking for TweetBiDAF Top-K spans.

The base QA model proposes spans.  A linear pairwise logistic model then
reorders them using fixed, interpretable features.  Validation gold answers
are never used to fit the reranker.
"""
import math
import re

import numpy as np
import torch

try:
    from . import data as D
    from .evaluate import normalize_answer, score_example
    from .train import to_device
except ImportError:
    import data as D
    from evaluate import normalize_answer, score_example
    from train import to_device


QUESTION_TYPES = ("what", "who", "how", "where", "why", "which", "when", "other")
REGIONS = ("body", "byline", "cross")
BASE_FEATURES = (
    "base_score", "score_margin", "reciprocal_rank", "start_logp", "end_logp",
    "length", "length_fraction", "start_fraction", "end_fraction", "distance_from_end",
    "question_overlap", "question_jaccard", "context_overlap", "context_jaccard",
    "starts_word", "ends_word", "contains_mention", "contains_url", "contains_number",
)
FEATURE_NAMES = (BASE_FEATURES
                 + tuple(f"qtype_{x}" for x in QUESTION_TYPES)
                 + tuple(f"region_{x}" for x in REGIONS))


def _tokens(text):
    return normalize_answer(text).split()


def _overlap(a, b):
    a, b = set(_tokens(a)), set(_tokens(b))
    if not a or not b:
        return 0.0, 0.0
    inter = len(a & b)
    return inter / len(a), inter / len(a | b)


def span_region(ex, start, end):
    boundary = ex["byline_from"]
    if boundary is None or end < boundary:
        return "body"
    if start >= boundary:
        return "byline"
    return "cross"


@torch.no_grad()
def topk_candidates(model, examples, device, top_k=10, batch_size=64, max_span_len=15):
    """Return Top-K legal spans and decomposed base scores for each example."""
    model.eval()
    records = []
    for batch in D.make_loader(examples, batch_size=batch_size):
        meta = batch["meta"]
        out = model(to_device(batch, device))
        start = torch.log_softmax(out["start_logits"], -1)
        end = torch.log_softmax(out["end_logits"], -1)
        padded = start.size(1)
        positions = torch.arange(padded, device=start.device)
        span_width = positions.unsqueeze(0) - positions.unsqueeze(1)
        legal_band = (span_width >= 0) & (span_width < max_span_len)
        combined = start.unsqueeze(2) + end.unsqueeze(1)
        combined = combined.masked_fill(~legal_band.unsqueeze(0), -1e30)
        for b, ex in enumerate(meta):
            length = len(ex["t_tokens"])
            legal_count = sum(min(max_span_len, length - i) for i in range(length))
            values, flat_indices = combined[b].reshape(-1).topk(min(top_k, legal_count))
            spans = []
            for value, flat in zip(values.tolist(), flat_indices.tolist()):
                i, j = flat // padded, flat % padded
                spans.append(dict(
                    start=i, end=j, text=D.span_text(ex, i, j),
                    start_logp=float(start[b, i]), end_logp=float(end[b, j]),
                    base_score=value, region=span_region(ex, i, j),
                ))
            records.append(dict(row_id=ex["row_id"], qid=ex["qid"], candidates=spans))
    return records


def candidate_features(ex, candidate, rank, top_score):
    """Fixed numeric feature vector; no corpus statistics or fitted parameters."""
    i, j = candidate["start"], candidate["end"]
    n = max(1, len(ex["t_tokens"])); width = j - i + 1
    question = ex["question"]; text = candidate["text"]
    left, right = max(0, i - 3), min(n - 1, j + 3)
    context = D.span_text(ex, left, right)
    q_overlap, q_jaccard = _overlap(question, text)
    c_overlap, c_jaccard = _overlap(question, context)
    raw_start = ex["t_tokens"][i]["raw"] if ex["t_tokens"] else ""
    raw_end = ex["t_tokens"][j]["raw"] if ex["t_tokens"] else ""
    values = [
        candidate["base_score"], candidate["base_score"] - top_score, 1.0 / rank,
        candidate["start_logp"], candidate["end_logp"], width, width / n,
        i / n, j / n, (n - 1 - j) / n,
        q_overlap, q_jaccard, c_overlap, c_jaccard,
        float(bool(re.search(r"\w", raw_start))), float(bool(re.search(r"\w", raw_end))),
        float("@" in text), float(bool(re.search(r"https?://|www\.", text, re.I))),
        float(bool(re.search(r"\d", text))),
    ]
    qtype = D._question_word(question)
    values.extend(float(qtype == x) for x in QUESTION_TYPES)
    values.extend(float(candidate["region"] == x) for x in REGIONS)
    return np.asarray(values, dtype=np.float64)


def feature_matrix(examples, records):
    by_id = {ex["row_id"]: ex for ex in examples}
    matrices = []
    for record in records:
        ex = by_id[record["row_id"]]
        top_score = record["candidates"][0]["base_score"]
        matrices.append(np.stack([candidate_features(ex, c, rank, top_score)
                                  for rank, c in enumerate(record["candidates"], 1)]))
    return matrices


def fit_pairwise_reranker(examples, records, regularization=1.0):
    """Fit one standardized linear ranking model from pairwise preferences.

    For each question, the candidate with maximum gold ROUGE-L is preferred to
    candidates with strictly lower ROUGE-L.  Reversed pairs balance the binary
    labels exactly.  `regularization` is fixed to 1.0 in the experiment.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    by_id = {ex["row_id"]: ex for ex in examples}
    matrices = feature_matrix(examples, records)
    pairs, labels = [], []
    useful = 0
    for record, matrix in zip(records, matrices):
        ex = by_id[record["row_id"]]
        quality = np.asarray([score_example(c["text"], ex["answers"], ("rougeL",))["rougeL"]
                              for c in record["candidates"]])
        best = int(quality.argmax())
        made_pair = False
        for other in range(len(quality)):
            if quality[best] <= quality[other] + 1e-12:
                continue
            diff = matrix[best] - matrix[other]
            pairs.extend((diff, -diff)); labels.extend((1, 0)); made_pair = True
        useful += int(made_pair)
    if not pairs:
        raise ValueError("no strict candidate preferences available for reranker training")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0 / regularization, solver="lbfgs", max_iter=1000,
                           random_state=4012),
    )
    model.fit(np.stack(pairs), np.asarray(labels))
    return model, dict(pair_rows=len(pairs), useful_questions=useful,
                       total_questions=len(records), regularization=regularization)


def fit_lambdamart_reranker(examples, records):
    """Fit one conservative LambdaMART model with a fixed shallow tree setup.

    Relevance is gold ROUGE-L quantised to five grades (0..4).  Query groups
    are passed explicitly, so the objective learns ordering within each
    question rather than a global answer-quality scale.
    """
    from lightgbm import LGBMRanker

    by_id = {ex["row_id"]: ex for ex in examples}
    matrices = feature_matrix(examples, records)
    labels, groups = [], []
    for record in records:
        ex = by_id[record["row_id"]]
        quality = [score_example(c["text"], ex["answers"], ("rougeL",))["rougeL"]
                   for c in record["candidates"]]
        labels.extend(min(4, int(round(4 * value))) for value in quality)
        groups.append(len(record["candidates"]))
    model = LGBMRanker(
        objective="lambdarank", metric="ndcg", n_estimators=150,
        learning_rate=0.05, num_leaves=15, max_depth=4,
        min_child_samples=50, reg_lambda=1.0,
        random_state=4012, deterministic=True, force_col_wise=True,
        verbosity=-1, n_jobs=-1,
    )
    model.fit(np.concatenate(matrices), np.asarray(labels, dtype=np.int32), group=groups)
    return model, dict(candidate_rows=len(labels), total_questions=len(records),
                       relevance_levels=5, fixed_trees=150, fixed_max_depth=4)


def rerank(model, examples, records):
    matrices = feature_matrix(examples, records)
    predictions, selections = [], []
    for record, matrix in zip(records, matrices):
        scores = (model.decision_function(matrix) if hasattr(model, "decision_function")
                  else model.predict(matrix))
        index = int(np.argmax(scores))
        predictions.append(record["candidates"][index]["text"])
        selections.append(dict(row_id=record["row_id"], qid=record["qid"],
                               selected_rank=index + 1, rerank_score=float(scores[index]),
                               **record["candidates"][index]))
    return predictions, selections


def linear_weights(model):
    """Return coefficients in standardized feature space for interpretation."""
    classifier = model.named_steps["logisticregression"]
    return sorted(zip(FEATURE_NAMES, classifier.coef_[0].tolist()),
                  key=lambda item: abs(item[1]), reverse=True)


def tree_importances(model):
    """Return fixed LambdaMART split importances for descriptive reporting."""
    return sorted(zip(FEATURE_NAMES, model.feature_importances_.tolist()),
                  key=lambda item: item[1], reverse=True)
