# Attention-analysis report

## Main finding
Across three validation seeds, TweetBiDAF v2 obtains ROUGE-L
44.22 ± 0.16.
Removing gated self-attention gives 42.41 ±
0.45, a reduction of 1.80 points.
The corresponding F1 reduction is 1.77 points.

## Interpretation
The controlled ablation supports retaining gated self-attention. Question-type
results suggest that explicit entity and location spans are easier than explanatory
`why` answers. Error sampling indicates that span boundaries, non-extractive gold
answers, and tweet-body versus byline confusion are the main remaining issues.

## Limitation
Attention weights are diagnostic visualisations, not causal explanations. Claims
about component utility should rely primarily on the repeated ablation result.
These are validation results; the locked test set was not accessed.
