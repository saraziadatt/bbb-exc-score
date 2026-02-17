#### Blood Brain Barrier Exclusion Score (BBBX)

##### To make predictions: 
1. Generate an sdf (with 3D structures) of your desired compounds
2. Edit the file path in cell 3, line 4 of the [bbb_exc_score script](./final_model/bbb_exc_score.ipynb)
3. Run the script to output your results. 

###### Google Colab: 
Alternatively you can use this [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com) to input smiles or an sdf file. Note that the smiles conversion to 3D structure in this notebook is different from the method used in the paper: 

##### Scripts used to generate the data presented in the paper:
1. [Dataset cleaning](./scripts/dataset_cleaning.ipynb)
2. [Dataset labelling](./scripts/dataset_labelling.ipynb)
3. [Training](./scripts/model_training.ipynb)
4. [Baseline model training](./scripts/baseline_model.ipynb)
4. [Model evaluation](./scripts/model_evaluation.ipynb)
5. [Deriving model rules](./scripts/derive_model_rules.ipynb)
6. [Evaluate other literature models](./evaluate_literature_models.ipynb)
7. [Compare all datasets and other models](./model_comparison.ipynb)
8. [Plot hyperparameter tuning and retrieve features](./scripts/hyperparameters.ipynb)
9. [Scraping scripts](./scripts/scrape_qsar_ann_model.ipynb)


##### All other data for the paper are available: 
- Datasets used are provided in the [datasets](./datasets) directory. 
- Models, training and hyperparameter tuning results are provided in the [model results](./model_results) directories, with each subdirectory indicating a different dataset or model 
- Performance evaluation results are provided in the [model evaluation](./model_evaluation) directory
- The final trained model and script to run new predictions is provided in the [final model](./final_model) directory 
