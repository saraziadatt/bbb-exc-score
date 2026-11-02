# Commands for the BBBX manuscript analyses

This file gives the workflow in order. The scripts and their outputs are under `scripts_and_results/`. The `datasets/` and `tools/` required to run these scripts are beside that directory.


## 2. Prepare the datasets and fixed folds

Skip to step 3 for reproduction from the processed descriptor files. Rebuilding the datasets follows:

1. Run `01_clean_label_generate_b3db_outerfolds.ipynb`
2. Generate the Butina assignments from the primary matrix using: 

```bash
python 02_generate_outerfold_butina.py \
  --input ../datasets/druglike_b3db_labelled_outerfolds.csv \
  --output ../datasets/druglike_b3db_labelled_butina_outerfolds.csv \
  --similarity-cutoff 0.6 \
  --radius 2 \
  --fp-size 2048 \
  --outer-folds 5 \
  --inner-folds 5 \
  --seed 15
```

3. Run `03_clean_label_generate_inxight.ipynb` to generate the balanced external set, and `03_clean_label_generate_inxight_unbalanced.ipynb` to generate the unbalanced external (Inxight) set. 

The expected inputs for the completed runs are:

| Variant | Training matrix | Compounds | Measured logBB |
|---|---|---:|---:|
| Random | `../datasets/druglike_b3db_labelled_outerfolds.csv` | 3,188 | 767 |
| Butina | `../datasets/druglike_b3db_labelled_butina_outerfolds.csv` | 3,188 | 767 |
| bRo5 | `../datasets/druglike_b3db_bRo5_labelled_outerfolds.csv` | 3,879 | 907 |

The balanced external file is `../datasets/inxight_labelled_balanced.csv` (210 compounds). The rule-analysis file is `../datasets/inxight_labelled_unbalanced.csv` (604 compounds). Nonpenetrants are class 1.

## 3. Train the descriptor and frozen-ImageMol models

To train all models used in the manuscript, run the following four commands for each of the variants (random_outerfolds, butina_outerfolds, random_outerfolds_bRo5) as listed. 

### random_outerfolds

```bash
python 04_train_custom_mtl_mse_ba_hpo.py \
  --input ../datasets/druglike_b3db_labelled_outerfolds.csv \
  --output-dir ./random_outerfolds/custom_mtl_bab_results \
  --mtl-trials 100 \
  --mtl-boost-rounds 1000 \
  --inner-folds 5 \
  --seed 15 \
  --logbb-threshold -1.0
```

```bash
python 05_train_logbb_pred_matched.py \
  --input ../datasets/druglike_b3db_labelled_outerfolds.csv \
  --output-dir ./random_outerfolds/logbb_pred_results \
  --tuning-mode published_ranges \
  --n-search-candidates 250 \
  --correlation-threshold 0.8 \
  --inner-folds 5 \
  --seed 15 \
  --n-jobs 4
```

```bash
python 06_train_imagemol_frozen_bbb.py \
  --input ../datasets/druglike_b3db_labelled_outerfolds.csv \
  --output-dir ./random_outerfolds/imagemol_frozen \
  --checkpoint ../tools/ImageMol.pth.tar \
  --resume-key state_dict \
  --trials 15 \
  --inner-folds 5 \
  --epochs 250 \
  --patience 25 \
  --batch-size 256 \
  --beta 2.5 \
  --margin-buffer 0.25 \
  --seed 42 \
  --device cpu
```

```bash
python 07_train_baselines_with_threshold_ablation.py \
  --input ../datasets/druglike_b3db_labelled_outerfolds.csv \
  --output-dir ./random_outerfolds/baseline_threshold_ablation_results \
  --lgbm-search-iterations 100 \
  --lgbm-boost-rounds 1000 \
  --inner-folds 5 \
  --seed 15 \
  --n-jobs 4 \
  --logbb-threshold -1.0
```

```bash
python 08_train_gnn_mtl_ablation.py \
  --input ../datasets/druglike_b3db_labelled_outerfolds.csv \
  --output-dir ./random_outerfolds/gnn_mtl_ablation_hpo_100 \
  --threshold -1.0 \
  --alpha 1.0 \
  --beta 2.5 \
  --margin-buffer 0.25 \
  --hidden-dim 128 \
  --layers 3 \
  --dropout 0.15 \
  --batch-size 128 \
  --learning-rate 0.0003 \
  --weight-decay 1e-05 \
  --epochs 100 \
  --patience 15 \
  --min-delta 0.0001 \
  --hpo-trials 100 \
  --hpo-epochs 75 \
  --hpo-patience 10 \
  --inner-folds 5 \
  --seed 15 \
  --num-workers 0
```

### butina_outerfolds

```bash
python 04_train_custom_mtl_mse_ba_hpo.py \
  --input ../datasets/druglike_b3db_labelled_butina_outerfolds.csv \
  --output-dir ./butina_outerfolds/custom_mtl_bab_results \
  --mtl-trials 100 \
  --mtl-boost-rounds 1000 \
  --inner-folds 5 \
  --seed 15 \
  --logbb-threshold -1.0
```

```bash
python 05_train_logbb_pred_matched.py \
  --input ../datasets/druglike_b3db_labelled_butina_outerfolds.csv \
  --output-dir ./butina_outerfolds/logbb_pred_results \
  --tuning-mode published_ranges \
  --n-search-candidates 250 \
  --correlation-threshold 0.8 \
  --inner-folds 5 \
  --seed 15 \
  --n-jobs 4
```

```bash
python 06_train_imagemol_frozen_bbb.py \
  --input ../datasets/druglike_b3db_labelled_butina_outerfolds.csv \
  --output-dir ./butina_outerfolds/imagemol_frozen \
  --checkpoint ../tools/ImageMol.pth.tar \
  --resume-key state_dict \
  --trials 15 \
  --inner-folds 5 \
  --epochs 250 \
  --patience 25 \
  --batch-size 256 \
  --beta 2.5 \
  --margin-buffer 0.25 \
  --seed 42 \
  --device cpu
```

```bash
python 07_train_baselines_with_threshold_ablation.py \
  --input ../datasets/druglike_b3db_labelled_butina_outerfolds.csv \
  --output-dir ./butina_outerfolds/baseline_threshold_ablation_results \
  --lgbm-search-iterations 100 \
  --lgbm-boost-rounds 1000 \
  --inner-folds 5 \
  --seed 15 \
  --n-jobs 4 \
  --logbb-threshold -1.0
```

### random_outerfolds_bRo5

```bash
python 04_train_custom_mtl_mse_ba_hpo.py \
  --input ../datasets/druglike_b3db_bRo5_labelled_outerfolds.csv \
  --output-dir ./random_outerfolds_bRo5/custom_mtl_bab_results \
  --mtl-trials 100 \
  --mtl-boost-rounds 1000 \
  --inner-folds 5 \
  --seed 15 \
  --logbb-threshold -1.0
```

```bash
python 05_train_logbb_pred_matched.py \
  --input ../datasets/druglike_b3db_bRo5_labelled_outerfolds.csv \
  --output-dir ./random_outerfolds_bRo5/logbb_pred_results \
  --tuning-mode published_ranges \
  --n-search-candidates 250 \
  --correlation-threshold 0.8 \
  --inner-folds 5 \
  --seed 15 \
  --n-jobs 4
```

```bash
python 06_train_imagemol_frozen_bbb.py \
  --input ../datasets/druglike_b3db_bRo5_labelled_outerfolds.csv \
  --output-dir ./random_outerfolds_bRo5/imagemol_frozen \
  --checkpoint ../tools/ImageMol.pth.tar \
  --resume-key state_dict \
  --trials 15 \
  --inner-folds 5 \
  --epochs 250 \
  --patience 25 \
  --batch-size 256 \
  --beta 2.5 \
  --margin-buffer 0.25 \
  --seed 42 \
  --device cpu
```

```bash
python 07_train_baselines_with_threshold_ablation.py \
  --input ../datasets/druglike_b3db_bRo5_labelled_outerfolds.csv \
  --output-dir ./random_outerfolds_bRo5/baseline_threshold_ablation_results \
  --lgbm-search-iterations 100 \
  --lgbm-boost-rounds 1000 \
  --inner-folds 5 \
  --seed 15 \
  --n-jobs 4 \
  --logbb-threshold -1.0
```

## 4. Collect each completed variant

Script 09 has no CLI arguments. Before each invocation, replace its existing `outerfolds_dir`, `model_dirs` and `out` assignments with the corresponding values below. Replace the existing later `out = ...` assignment too; adding a second assignment at the top would be overwritten. Do not change the collection logic.

### Random folds

```bash
python 09_combine_all_results.py \
  --outerfolds-dir ./random_outerfolds \
  --output-dir ./random_outerfolds/all_model_results_main \
  --gnn-dir ./random_outerfolds/gnn_mtl_ablation_hpo
```

### Butina folds

```bash
python 09_combine_all_results.py \
  --outerfolds-dir ./butina_outerfolds \
  --output-dir ./butina_outerfolds/all_model_results
```

### bRo5 folds

```bash
python 09_combine_all_results.py \
  --outerfolds-dir ./random_outerfolds_bRo5 \
  --output-dir ./random_outerfolds_bRo5/all_model_results
```

## 5. Evaluate the outer-fold predictions

```bash
python 10_evaluation_and_significance_testing.py \
  --input-format oof \
  --predictions-file ./random_outerfolds/all_model_results_main/all_models_predictions.csv \
  --output-dir ./random_outerfolds/all_model_eval_main \
  --bootstrap 10000 \
  --permutations 10000 \
  --confidence 0.95 \
  --seed 20260730 \
  --chunk-size 250
```

```bash
python 10_evaluation_and_significance_testing.py \
  --input-format oof \
  --predictions-file ./butina_outerfolds/all_model_results/all_models_predictions.csv \
  --output-dir ./butina_outerfolds/all_model_eval \
  --bootstrap 10000 \
  --permutations 10000 \
  --confidence 0.95 \
  --seed 20260730 \
  --chunk-size 250
```

```bash
python 10_evaluation_and_significance_testing.py \
  --input-format oof \
  --predictions-file ./random_outerfolds_bRo5/all_model_results/all_models_predictions.csv \
  --output-dir ./random_outerfolds_bRo5/all_model_eval \
  --bootstrap 10000 \
  --permutations 10000 \
  --confidence 0.95 \
  --seed 20260730 \
  --chunk-size 250
```

## 6. Generate and evaluate external Inxight predictions

### Prediction

```bash
python 11_predict_external_set_threshold_aware.py \
  --external-csv ../datasets/inxight_labelled_balanced.csv \
  --model-dir ./random_outerfolds/all_model_results_main/fitted_models \
  --models imagemol_frozen__imagemol_resnet18_frozen_custom_mtl__outer,imagemol_frozen__imagemol_resnet18_frozen_regression__outer,custom_mtl_bab,dummy_mean_regression,dummy_prior_classification,lightgbm_mse_regression,lightgbm_mse_regression_threshold,logbb_pred_shaker_retrained,gnn_custom_mtl,gnn_mse \
  --output-dir ./inxight_results/random_outerfolds \
  --smiles-col SMILES \
  --imagemol-checkpoint ../tools/ImageMol.pth.tar \
  --imagemol-resume-key state_dict \
  --imagemol-workers 0 \
  --pretrained-device cpu \
  --pretrained-batch-size 64
```

```bash
python 11_predict_external_set_threshold_aware.py \
  --external-csv ../datasets/inxight_labelled_balanced.csv \
  --model-dir ./butina_outerfolds/all_model_results/fitted_models \
  --models imagemol_frozen__imagemol_resnet18_frozen_custom_mtl__outer,imagemol_frozen__imagemol_resnet18_frozen_regression__outer,custom_mtl_bab,dummy_mean_regression,dummy_prior_classification,lightgbm_mse_regression,lightgbm_mse_regression_threshold,logbb_pred_shaker_retrained \
  --output-dir ./inxight_results/butina_outerfolds
```

```bash
python 11_predict_external_set_threshold_aware.py \
  --external-csv ../datasets/inxight_labelled_balanced.csv \
  --model-dir ./random_outerfolds_bRo5/all_model_results/fitted_models \
  --models imagemol_frozen__imagemol_resnet18_frozen_custom_mtl__outer,imagemol_frozen__imagemol_resnet18_frozen_regression__outer,custom_mtl_bab,dummy_mean_regression,dummy_prior_classification,lightgbm_mse_regression,lightgbm_mse_regression_threshold,logbb_pred_shaker_retrained \
  --output-dir ./inxight_results/random_outerfolds_bRo5
```

### External permutation statistics

```bash
python 10_evaluation_and_significance_testing.py \
  --input-format external \
  --predictions-file ./inxight_results/random_outerfolds/all_selected_models.csv \
  --output-dir ./inxight_results/random_outerfolds/paired_permutation_results \
  --bootstrap 10000 \
  --permutations 10000 \
  --confidence 0.95 \
  --seed 20260730 \
  --chunk-size 250
```

```bash
python 10_evaluation_and_significance_testing.py \
  --input-format external \
  --predictions-file ./inxight_results/butina_outerfolds/all_selected_models.csv \
  --output-dir ./inxight_results/butina_outerfolds/paired_permutation_results \
  --bootstrap 10000 \
  --permutations 10000 \
  --confidence 0.95 \
  --seed 20260730 \
  --chunk-size 250
```

```bash
python 10_evaluation_and_significance_testing.py \
  --input-format external \
  --predictions-file ./inxight_results/random_outerfolds_bRo5/all_selected_models.csv \
  --output-dir ./inxight_results/random_outerfolds_bRo5/paired_permutation_results \
  --bootstrap 10000 \
  --permutations 10000 \
  --confidence 0.95 \
  --seed 20260730 \
  --chunk-size 250
```

## 7. Run the fixed-parameter beta sweep

The completed sweep reuses each final custom model's selected parameters and seed. The recorded beta grid is:

```bash
python 12_beta_sensitivity_analysis.py \
  --input ../datasets/druglike_b3db_labelled_outerfolds.csv \
  --reference-parameters ./random_outerfolds/custom_mtl_bab_results/custom_mtl_best_parameters.csv \
  --reference-manifest ./random_outerfolds/custom_mtl_bab_results/run_manifest.json \
  --output-dir ./results/beta_sensitivity_final_reference \
  --beta-values 0,0.25,0.5,1,2,2.5,3,5,10,20,30
```
Run the model evaluation notebook (`13_model_evaluation.ipynb`) to generate: 

1. `figure_s1` and `results/score_peak_separation/`.
2. `figure_1`, using `random_outerfolds/all_model_results_main/all_models_predictions.csv` and the saved beta sweep.
3. `figure_s2` threshold sweeps.
4. Outer-fold score calibration and `figure_s3`; retain `INNER_OOF_PATH = None`, matching the recorded outer-training in-sample calibration maps applied to held-out compounds.
5. The two Inxight calibration outputs, first `custom_mtl_bab`, then `lightgbm_mse_regression_threshold`.
6. The combined Figure 2 export, after both Inxight mapping outputs exist.
7. The paired calibration comparison: ten bins, 50,000 permutations, 10,000 bootstrap draws and seed 15, as set in the supplied comparison cell.

The saved model-comparison tables come from script 10, not the notebook's exploratory metric calculations. The final assembled Figure 2 still needs to be included in the release; it was absent from the supplied figure archive.

## 8. Generate descriptor clusters and grouped SHAP

Use the Butina matrix and its own fitted custom models:

```bash
python 14_generate_feature_clusters_only.py \
  --input ../datasets/druglike_b3db_labelled_butina_outerfolds.csv \
  --output-dir ./results/shap_analysis/butina_outerfolds \
  --correlation-threshold 0.8 \
  --correlation-method spearman \
  --linkage-method complete \
  --min-nonmissing 20
```

```bash
python 15_calculate_grouped_shap_full_models.py \
  --input ../datasets/druglike_b3db_labelled_butina_outerfolds.csv \
  --models custom_mtl_bab \
  --model-dirs ./butina_outerfolds/custom_mtl_bab_results \
  --cluster-results-dir ./results/shap_analysis/butina_outerfolds \
  --output-dir ./results/shap_analysis/butina_outerfolds/grouped_shap \
  --top-n 10 \
  --multiple-testing bh \
  --save-local-npz \
  --save-local-top-n 20
```

Run the rule deriving notebook (`16_deriving_model_rules.ipynb`) to generate figures 3 and 4. 

## 9. Run the additional evaluation branches

### Kp,uu: prediction, then statistics


```bash
python 17_kpuu_transfer.py \
  --development-csv ../datasets/druglike_b3db_labelled_outerfolds.csv \
  --model-dir ./random_outerfolds/all_model_results_main/fitted_models \
  --models lightgbm_mse_regression,custom_mtl_bab,lightgbm_mse_regression_threshold \
  --cache-dir ./kpuu_transfer/kpuu_downloads \
  --output-dir ./kpuu_transfer \
  --openchemlib-jar ../tools/datawarrior_all.jar \
  --conformer-seed 20260915 \
  --dielectric-constant 1 \
  --threads 1 \
  --species rat \
  --bootstrap 2000 \
  --seed 42
```

```bash
python 18_kpuu_statistics.py \
  --predictions-csv ./kpuu_transfer/transfer_predictions.csv \
  --output-dir ./kpuu_transfer/kpuu_statistics \
  --kpuu-cutoff 0.3 \
  --bootstrap 2000 \
  --permutations 10000 \
  --seed 42 \
  --batch-size 256
```

### hERG: training, then statistics


```bash
python 19_herg_mtl_vs_mse.py \
  --output-dir ./herg_mtl_vs_mse \
  --cache-dir ./herg_mtl_vs_mse/herg_downloads \
  --screen-sign tdc-negative \
  --max-ic50-range 1.0 \
  --max-screen 0 \
  --outer-folds 5 \
  --inner-folds 3 \
  --split random \
  --trials 30 \
  --beta-min 0.5 \
  --beta-max 5000 \
  --rounds 500 \
  --bits 2048 \
  --threads 4 \
  --seed 42
```

```bash
python 20_herg_statistics.py \
  --predictions-csv ./herg_mtl_vs_mse/predictions.csv \
  --reference mse \
  --comparison mtl \
  --threshold 5.0 \
  --bootstrap 2000 \
  --permutations 2000 \
  --seed 42
```

### Released ImageMol BBBP classifier

```bash
python 21_filter_inxight_bbbp.py \
  --inxight-csv ../datasets/inxight_labelled_balanced.csv \
  --bbbp-csv ../datasets/BBBP.csv \
  --smiles-col SMILES \
  --bbbp-smiles-col smiles \
  --expected-rows 210 \
  --output ../datasets/inxight_without_bbbp.csv
```

```bash
python 22_evaluate_imagemol_bbbp_external.py \
  --input ../datasets/inxight_without_bbbp.csv \
  --checkpoint ../tools/bbbp.pth \
  --output-dir ./inxight_results/random_outerfolds \
  --smiles-col SMILES \
  --label-col bbb_class \
  --penetrant-label 0 \
  --nonpenetrant-label 1 \
  --batch-size 16 \
  --device cpu
```

### Hessian diagnostics

```bash
python 24_check_loss_hessian.py
```

## 10. Calibration, rules and figure notebooks



### Notebook 16: interpretation and rules

Use these sources:

| Setting | Value |
|---|---|
| Training matrix | `../datasets/druglike_b3db_labelled_butina_outerfolds.csv` |
| Model directory | `./butina_outerfolds/all_model_results_main/fitted_models` |
| Grouped SHAP summary | `./results/shap_analysis/butina_outerfolds/grouped_shap/grouped_shap_summary.csv` |
| External rule dataset | `../datasets/inxight_labelled_unbalanced.csv` |
| Selected outer fold | `0` |
| Saved inner-fold column | `inner_fold_outer_0` |

Run its training-partition feature/rule calculations, held-out and external rule evaluation, and the Figure 3/4 and supplementary interpretation exports in notebook order. Retain `FIGURE_SOURCES['flowchart'] = None`: **notebook 16 is not the Figure 5 builder**. Figure 5 uses the cleaning audit from notebook 01 and the existing figure-preparation step. No standalone Figure 5 generation command is recorded in the supplied files.

## 12. Assemble the manuscript tables last

Open `24_manuscript_and_supplementary_tables.ipynb` once all required numerical sources exist. Set or retain:

| Setting | Value |
|---|---|
| `variant` | `"random_outerfolds"` |
| `PROJECT_DIR` | Directory containing the three variant folders (`scripts_and_results/`) |
| Primary OOF statistics | `random_outerfolds/all_model_eval_main/` |
| Random external statistics | `inxight_results/random_outerfolds/paired_permutation_results/` |
| Butina OOF statistics | `butina_outerfolds/all_model_eval/` |
| bRo5 OOF statistics | `random_outerfolds_bRo5/all_model_eval/` |
| bRo5 external statistics | `inxight_results/random_outerfolds_bRo5/paired_permutation_results/` |
| `KPUU_STATISTICS_DIR` | `kpuu_transfer/kpuu_statistics/` |
| `HERG_STATISTICS_DIR` | `herg_mtl_vs_mse/statistics/` |
| `BBBP_METRICS_FILE` | `inxight_results/random_outerfolds/bbbp_metrics.csv` |
| `DESCRIPTOR_BEFORE_FILE` | Original unfiltered training feature matrix/list exported by the notebook-01 correction; its new path is not recorded in the supplied archive. |

Keep the explicit Inxight ImageMol model-ID mappings and the BBBP average-precision column mapping already present in the notebook. Preserve `TABLES_TO_SKIP = {"main_06", "supplementary_01", "supplementary_17"}` for the same generated-table set. Run the configured table-generation cells and inspect `table_manifest.csv`, `source_loading_report.csv` and `missing_statistics.csv`.

The output directory is `results/manuscript_tables/random_outerfolds_paper_numbered/`. Notebook 24 formats saved numerical results; it does not rerun model training or the statistical tests.

## Provenance and remaining gaps in the historical command record

| Command group | Primary evidence |
|---|---|
| 04, 05 and 07 | Each variant's training `run_manifest.json` |
| 06 | Each `imagemol_frozen/manifest.json` and the supplied script defaults |
| 08 | The selected random 100-trial and bRo5 GNN `run_manifest.json` argument dictionaries |
| 09 | Supplied collector source, collected outputs and training/collected file identity checks |
| 10 | Each `analysis_metadata.json`; successful exact replay with chunk size 250 |
| 11 | Each external `prediction_manifest.json` |
| 12 | Butina/bRo5 `01_eval/05_evaluation_config.json`; source-default reconstruction for the random aligned export |
| 13 | Beta-sweep `run_manifest.json` and referenced final-parameter file |
| 15–16 | Cluster and grouped-SHAP manifests; explicit relocation to `results/shap_analysis/` |
| 18–21 | Native run/statistics manifests and saved source caches |
| 22–23 | Supplied filtering script, exclusion audit and retained BBBP metadata |
| 24 | `diagnostic_manifest.json` and supplied editable constants |
| Notebooks | Supplied configuration/code and corresponding saved outputs |

The archives do not establish the exact shell spelling/order, environment variables, notebook click order, or every unrecorded default. Additional historical ImageMol-only external invocations for Butina/bRo5 and the standalone Figure 5 creation step are not reconstructed as if they were logged. The output-directory normalizations above are explicit; they preserve the intended inputs/settings while matching the planned release layout.
