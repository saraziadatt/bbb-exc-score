# Blood Brain Barrier Exclusion (BBBX) Score 

This repository contains the trained Blood Brain Barrier Exclusion Score (BBBX) model and all scripts used to generate the data and analyses presented in the associated manuscript.

---

## Making Predictions on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saraziadatt/bbb-exc-score/blob/main/colab_notebook/bbb_exc_score_colab_v4.ipynb)
<br>
Use the Colab notebook to enter SMILES directly, upload a CSV containing SMILES, upload a prepared 3D SDF file, or reuse a saved descriptor CSV.  

---

## Making Predictions Locally

### 1. Set Up the Environment

The [environment.yml](./environment.yml) file specifies the dependencies for this workflow.

Preparing 3D structures from SMILES requires a Java Development Kit providing both java and javac, together with tools/datawarrior_all.jar. 

##### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate bbbx_env
```

##### pip

```bash
pip install -r requirements.txt
```

### 2. Run the Prediction Notebook

Open: [final_model/bbb_exc_score.ipynb](./final_model/bbb_exc_score_predictions.ipynb)

In the "Select input" cell, choose input_kind: `SMILES`, `CSV`, `3D SDF`, or `Descriptor CSV`. For file inputs, set `file` to your input file. For direct SMILES input, edit `smiles_space_separated`. For a SMILES CSV, set `smiles_column` to the appropriate column name. Predictions are outputted in the final cell and are saved as a `CSV` in `final_model/`

The output includes predicted logBB, the BBBX score, the predicted BBB class, and processing or training-set selection flags. Compounds outside the training-sett selection criteria still receive predictionsn when processing succeeds. Invalid or uprocessable inputs remain in the output with the corresponding error message. 

---

## Reproducing the Manuscript Analyses

The workflow can be rerun from the supplied starting datasets using the scripts below. To reproduce the reported numerical results, use the supplied processed datasets, saved descriptors, fold assignments, and corresponding model and prediction files. Regenerating descriptors or retraining models can introduce numerical differences because of the 3D structure generation. 

See [00_RUN_COMMANDS.md](./scripts_and_results/00_RUN_COMMANDS.md) for the execution order, commands, and run settings.

## Scripts Used for the Manuscript

The analysis scripts and notebooks are located in [scripts_and_results/](./scripts_and_results/).

| Analysis | Scripts and notebooks |
|---|---|
| Dataset preparation | [01: B3DB cleaning, labelling, descriptors, and random folds](./scripts_and_results/01_clean_label_generate_b3db_outerfolds.ipynb); [02: Butina cluster-separated folds](./scripts_and_results/02_generate_outerfold_butina.py); [03: balanced Inxight dataset](./scripts_and_results/03_clean_label_generate_inxight.ipynb); [03: unbalanced Inxight dataset](./scripts_and_results/03_clean_label_generate_inxight_unbalanced.ipynb) |
| Model training | [04: custom multitask LightGBM](./scripts_and_results/04_train_custom_mtl_mse_ba_hpo.py); [05: matched retraining of logBB_pred](./scripts_and_results/05_train_logbb_pred_matched.py); [06: frozen ImageMol models](./scripts_and_results/06_train_imagemol_frozen_bbb.py); [07: baseline models and threshold tuning](./scripts_and_results/07_train_baselines_with_threshold_ablation.py); [08: GNN loss comparison](./scripts_and_results/08_train_gnn_mtl_ablation.py) |
| Result collection and statistical comparisons | [09: collect model outputs](./scripts_and_results/09_combine_all_results.py); [10: performance evaluation and paired permutation tests](./scripts_and_results/10_evaluation_and_significance_testing.py) |
| External validation | [11: generate Inxight predictions](./scripts_and_results/11_predict_external_set_threshold_aware.py) |
| Sensitivity, calibration, and performance figures | [12: sensitivity to the classification-loss weight](./scripts_and_results/12_beta_sensitivity_analysis.py); [13: model evaluation, calibration, and figures](./scripts_and_results/13_model_evaluation.ipynb) |
| Model interpretation | [14: descriptor clustering](./scripts_and_results/14_generate_feature_clusters_only.py); [15: grouped SHAP analysis](./scripts_and_results/15_calculate_grouped_shap_full_models.py); [16: rule derivation, validation, and interpretation figures](./scripts_and_results/16_deriving_model_rules.ipynb) |
| Kp,uu transfer evaluation | [17: dataset preparation and transfer predictions](./scripts_and_results/17_kpuu_transfer.py); [18: statistical analysis](./scripts_and_results/18_kpuu_statistics.py) |
| hERG benchmark | [19: dataset preparation and multitask-versus-MSE comparison](./scripts_and_results/19_herg_mtl_vs_mse.py); [20: statistical analysis](./scripts_and_results/20_herg_statistics.py) |
| Released ImageMol BBBP classifier | [21: remove compounds overlapping with BBBP](./scripts_and_results/21_filter_inxight_bbbp.py); [22: external classifier evaluation](./scripts_and_results/22_evaluate_imagemol_bbbp_external.py) |
| Loss-function diagnostics | [23: analytic and stored leaf Hessian checks](./scripts_and_results/23_check_loss_hessian.py) |
| Manuscript tables | [24: assemble and export main and supplementary tables](./scripts_and_results/24_manuscript_and_supplementary_tables.ipynb) |


## Repository Structure

- [`datasets/`](./datasets/) — Starting datasets, processed descriptor matrices, labels, and fixed fold assignments.
- [`final_model/`](./final_model/) — Saved BBBX models, preprocessing information, and the local prediction notebook.
- [`colab_notebook/`](./colab_notebook/) — Google Colab prediction notebook.
- [`scripts_and_results/`](./scripts_and_results/) — Manuscript analysis scripts, notebooks, and saved outputs.
  - [`random_outerfolds/`](./scripts_and_results/random_outerfolds/) — Primary random-fold training and evaluation results.
  - [`butina_outerfolds/`](./scripts_and_results/butina_outerfolds/) — Butina cluster-separated training and evaluation results.
  - [`random_outerfolds_bRo5/`](./scripts_and_results/random_outerfolds_bRo5/) — Training and evaluation with expanded druglike criteria.
  - [`inxight_results/`](./scripts_and_results/inxight_results/) — External Inxight predictions and evaluations.
  - [`shap_analysis/`](./scripts_and_results/shap_analysis/) — Descriptor clusters and grouped SHAP outputs.
  - [`kpuu_transfer/`](./scripts_and_results/kpuu_transfer/) — Kp,uu transfer predictions and statistical analyses.
  - [`herg_mtl_vs_mse/`](./scripts_and_results/herg_mtl_vs_mse/) — hERG benchmark data, models, predictions, and statistics.
  - [`results/`](./scripts_and_results/results/) — Additional analysis outputs and compiled manuscript tables.
  - [`figures/`](./scripts_and_results/figures/) — Exported manuscript and supplementary figures.
- [`tools/`](./tools/) — Structure-preparation tools and pretrained checkpoints.
- [`00_RUN_COMMANDS.md`](./scripts_and_results/00_RUN_COMMANDS.md) — Ordered commands, run settings, and notebook configuration instructions.


## Citation

If you use this model, please cite the associated manuscript:
Ziadat, S., Noden, M., Yu, L., Weaver, D. F. & Reed, M. A. Blood-Brain Barrier Exclusion (BBBX) Score: Designing Drugs Not to Enter the Brain. ChemRxiv. 2026. 
