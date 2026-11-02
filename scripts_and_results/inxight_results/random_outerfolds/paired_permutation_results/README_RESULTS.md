# Complete paired model comparisons

All 45 unique named model pairs are exported, including native/tuned
and pretrained variants with identical continuous predictions. Each unordered pair
is stored once. `difference_a_minus_b` and its CI always refer to model A minus B.
Reversing a comparison negates the difference and reverses/negates CI endpoints;
the two-sided p-value stays unchanged.

P-values use 10,000 within-compound paired prediction-swap permutations
with the finite-simulation correction (extreme + 1)/(valid + 1). Model outputs move
together. Class labels and evaluation populations never change. No thresholds are
fit on evaluation data. Holm correction covers all finite metrics within each
model pair/subset, including the new distance metrics. Consequently, the new
adjusted p-values may differ from older runs with smaller metric families.

10,000 shared paired bootstrap draws, stratified by BBB class (external compounds),
give 95.0% percentile CIs. Pooled `estimate_ci_*` columns describe an
individual model; pairwise `ci_lower/ci_upper` describe A minus B. The manuscript
notebook uses the pairwise intervals. Arithmetic and saved differences use float64.

Distance is mean |predicted logBB + 1| among incorrectly classified compounds, using each model's actual saved decisions (including tuned thresholds). The margin is fixed at -1; classification uses the table's classification population and regression uses measured-logBB compounds. Means with no mistakes or no predicted logBB are undefined.
`wrong_side_count` counts all classification errors (FP + FN).
`np_wrong_side_count` counts true non-penetrants predicted as penetrant (FN).
Both counts use each model's actual saved decisions, including tuned thresholds;
both receive estimate/difference CIs and paired permutation tests.
The NP/penetrant distance metrics restrict those mistakes by true class. Different
models can have different sets of mistakes. Mean distance is descriptive of those
sets; it should be read together with the mistake count.

Undefined metric replicates are excluded from that metric's resampling distribution
and flagged `conditional_on_defined_replicates`; their valid counts are exported.
No logBB prediction, constant-input Pearson correlation, or no mistakes can make
a metric and its comparison undefined. Such cases remain NA with an explicit
reason. Identical defined outputs yield delta 0, CI (0, 0), and p=1.

Files:
- `pooled_metrics_selected_models_long.csv`: every model/subset/metric and estimate CI.
- `paired_two_sided_permutation_holm_by_pair_and_dataset.csv`: all pairwise difference
  CIs, raw p-values, and `permutation_p_two_sided_holm_pair_dataset`.
- `pairwise_coverage_audit.csv`: complete metric-row and finite-statistic counts.
- `unavailable_pairwise_statistics.csv`: exact reasons for remaining unavailable tests.
- `subset_input_audit.csv`: evaluated or empty populations; no compounds are filtered.

For external predictions run this same script with `--input-format external` and
`--predictions-file path/to/01_all_external_predictions.csv`, choosing a separate
`--output-dir`. Predictions must already be aggregated to one row per model/compound
and scored so that higher score_np means NP. Optional experimental `true_logBB`
enables external regression; class labels alone cannot supply regression statistics.

Run notebook 19 with INPUT_MODE='results' and format='09' for each updated output
folder. An existing compiled CSV contains old statistics and must be regenerated.
The notebook retains custom MTL versus the better native MSE/logBB_pred comparator
per metric and places tuned MSE versus repeated custom MTL in separate tables.
The original per-pair Holm procedure does not correct across all model pairs or
for choosing the better comparator using these evaluation results.
