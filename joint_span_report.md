# Joint-span objective experiment

In a one-seed controlled comparison, quality weighting alone changed ROUGE-L by
+1.63, joint span scoring alone by -0.34,
and the complete joint + quality + hard-negative configuration by
+0.47 relative to the current objective.

These validation-only results isolate each component under the same split, seed, and frozen
hyperparameters. Any selected configuration should be confirmed over three seeds before use.
The locked test set was not accessed.
