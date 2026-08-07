# Paired permutation significance results

The only hypothesis test in this analysis is a paired, two-sided permutation
 test based on randomly exchanging the two models' predictions within each
 compound. Holm correction is applied separately across the five selected
 regression metrics or 9 selected classification metrics within each unique
 model pair and evaluation dataset.

Paired bootstrap resampling stratified by original outer fold × BBB class is
 used only for the 95.0% percentile confidence intervals.

Primary output:
- `paired_two_sided_permutation_holm_by_pair_and_dataset.csv`

Primary adjusted p-value column:
- `permutation_p_two_sided_holm_pair_dataset`
