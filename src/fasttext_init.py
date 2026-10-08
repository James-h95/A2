"""Optional in-domain word vectors: FastText (gensim) trained from scratch on the
*training split's* tweets and questions only. No external text and no
pretrained weights, so the main model is still trained from scratch on
TweetQA alone."""
import torch


def fasttext_vectors(wv, cfg, train_examples, epochs=20, seed=1):
    from gensim.models import FastText
    sents, seen = [], set()
    for ex in train_examples:
        if ex["tweet"] not in seen:                 # each tweet once
            seen.add(ex["tweet"])
            sents.append([t["word"] for t in ex["t_tokens"]])
        sents.append([t["word"] for t in ex["q_tokens"]])
    ft = FastText(vector_size=cfg["word_dim"], window=5, min_count=1, sg=1, epochs=epochs,
                  seed=seed, workers=1, min_n=3, max_n=5)
    ft.build_vocab(sents)
    ft.train(sents, total_examples=len(sents), epochs=epochs)
    W = torch.randn(len(wv), cfg["word_dim"]) * 0.1
    W[0] = 0.0
    for i, w in enumerate(wv.itos[2:], start=2):
        W[i] = torch.tensor(ft.wv[w])               # subword n-grams also give OOV-safe vectors
    return W
