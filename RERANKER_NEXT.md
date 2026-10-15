# Step 11: OOF confirmation, ensemble, feature study

Open `step11_colab_reranker_confirmation_ensemble_features.ipynb` in Colab and run top to bottom with a GPU runtime.

Required inputs:

- `data/train.csv` (upload as `train.csv`)
- `attention_variants_artifacts.zip` (exact Step 10 source checkpoint)
- `a2_v2_colab_artifacts.zip` (matched v2 seeds 1–3)
- `oof_span_reranker_artifacts.zip` (saved fixed LambdaMART)
- Drive folder `MyDrive/CITS4012_A2_reranker_oof` containing `fold1_oof_top10.jsonl`, `fold2_oof_top10.jsonl`, and `fold3_oof_top10.jsonl` from Step 10. These caches are not included in its compact ZIP. If absent, restore them or rerun Step 10; steps 1–2 still run without them.

Execution order:

1. Reproduce exact CurrentGated seed-1 results, then evaluate the unchanged saved reranker on each v2 seed. Report each delta separately; CurrentGated is not relabelled as a member of the matched v2 three-seed family.
2. Average start/end probabilities across v2 seeds with each checkpoint's own vocabulary. Evaluate Top-1 and fixed reranker transfer. Transfer is exploratory because the original OOF training candidates came from single models.
3. Compare original features against eight fixed boundary/confidence features on a tweet-grouped internal development partition of existing OOF records. Keep tree settings unchanged.
4. Fit both predefined variants on all OOF records and evaluate on the same exact CurrentGated validation candidates. Save models, feature importance, scores and per-question predictions.

Outputs are saved after each stage to `MyDrive/CITS4012_A2_reranker_next/reranker_next_artifacts.zip`. No base model is retrained and no test data is loaded. Do not tune new feature choices from validation scores.

The internal development check reuses existing OOF records and is exploratory: their generating models can have seen other internal-dev examples. A fully independent selection protocol requires nested grouped candidate generation. Matched ensemble OOF training would likewise require further fold-model inference/training and is not claimed by this notebook.

Local verification: all embedded Python cells compile; synthetic checks cover added boundary indicators, candidate overlap support, determinism and finite values. Colab inference and LightGBM fitting remain to be run.
