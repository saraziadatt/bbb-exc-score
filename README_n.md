## Blood Brain Barrier Exclusion Score (BBBX)

### Overview

This repository contains the trained Blood Brain Barrier Exclusion Score (BBBX) model and all scripts used to generate the data and analyses presented in the associated manuscript.

The BBBX model predicts blood–brain barrier exclusion properties from molecular structure.

---

### Making Predictions

#### 1. Prepare Input

Generate an `.sdf` file containing 3D molecular structures of the compounds you wish to evaluate.

---

#### 2. Set Up the Environment

Python 3.9–3.11 is required. Python 3.12+ is not currently

##### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate bbb_env
```

##### pip

```bash
pip install -r requirements.txt
```

---

#### 3. Run the Prediction Notebook

Open: [final_model/bbb_exc_score.ipynb](./final_model/bbb_exc_score.ipynb)

Edit the file path in Cell 3 to point to your `.sdf` file, then run the notebook to generate predictions.

---

### Google Colab Version
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saraziadatt/bbb-exc-score/blob/main/colab_notebooks/bbb_exc_score_colab_V0_3.ipynb)
<br>
Alternatively you can use this colab notebook to input smiles or an sdf file. Note that the smiles conversion to 3D structure in this notebook is different from the method used in the paper. 

---

### Scripts Used for the Manuscript

1. [Dataset cleaning](./scripts/dataset_cleaning.ipynb)
2. [Dataset labelling](./scripts/dataset_labelling.ipynb)
3. [Training](./scripts/model_training.ipynb)
4. [Baseline model training](./scripts/baseline_model.ipynb)
4. [Model evaluation](./scripts/model_evaluation.ipynb)
5. [Deriving model rules](./scripts/derive_model_rules.ipynb)
6. [Evaluate other literature models](./evaluate_literature_models.ipynb)
7. [Compare all datasets and other models](./model_comparison.ipynb)
8. [Plot hyperparameter tuning and retrieve features](./scripts/hyperparameters.ipynb)
9. [Data scraping scripts](./scripts/scrape_qsar_ann_model.ipynb)


---

### Repository Structure
- [`datasets/`](./datasets) — Datasets used in the study  
- [`model_results/`](./model_results) — Training and hyperparameter optimization outputs  
- [`model_evaluation/`](./model_evaluation)  — Performance evaluation results  
- [`final_model/`](./final_model) — Final trained model and prediction notebook  

---

### Citation

If you use this model, please cite the associated manuscript.

Citation information will be added upon publication.
