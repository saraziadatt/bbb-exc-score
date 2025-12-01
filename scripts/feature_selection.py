# Import libraries
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors


# Incorporating fingerprint similarity function 
# Other feature selection techniques (RFE) are not included as these are very easy to code into the notebook. 

# Function to calculate the strcuture similarity (Tanimoto Similarity) based on the fingerprint. 

def mw_diff(smiles): 
    
    smiles = np.array(smiles)

    all_mols = []

    for smile_id, smile in enumerate(smiles): 
        # print(smile_id)
        mol = Chem.MolFromSmiles(smile)
        all_mols.append(mol)

    # use a triangular matrix: 
    mw_diff_lst = []
    for mol_id in range(len(all_mols)): 
        for mol_id_second_mol in range(len(all_mols)): 
            mw_diff = Descriptors.MolWt(all_mols[mol_id])-Descriptors.MolWt(all_mols[mol_id_second_mol])
            mw_diff_lst.append(mw_diff)
            
    # mw_diff_matrix = squareform(np.array(mw_diff_lst))
    # np.fill_diagonal(mw_diff_matrix, 1)
    
    return mw_diff_lst


def similarity_train_test(train_smiles, test_smiles):

    train_smiles = np.array(train_smiles)
    test_smiles = np.array(test_smiles)

    all_train_mols, all_test_mols = [], []

    for smile in train_smiles: 
        # print(smile_id)
        mol = Chem.MolFromSmiles(smile)
        all_train_mols.append(mol)

    for smile in test_smiles: 
        # print(smile_id)
        mol = Chem.MolFromSmiles(smile)
        all_test_mols.append(mol)


    # generate fingerprints 
    fpgen = AllChem.GetRDKitFPGenerator()

    train_fps, test_fps = [],[]
    
    for mol_id, train_mol in enumerate(all_train_mols):
        try: 
            train_fps.append(fpgen.GetFingerprint(train_mol))
        except: 
            print('invalid:', mol_id)

    for mol_id, test_mol in enumerate(all_test_mols):
        try: 
            test_fps.append(fpgen.GetFingerprint(test_mol))
        except: 
            print('invalid:', mol_id)


    sim_vals, max_sim = [], []

    # use a triangular matrix: 
    for fp_id in range(len(test_fps)): 
        for fp_id_second_mol in range(len(train_fps)): 
            sim_vals.append(DataStructs.TanimotoSimilarity(test_fps[fp_id],train_fps[fp_id_second_mol]))
        max_sim.append(np.max(sim_vals))
    # sim_matrix = squareform(np.array(sim_vals))
    # np.fill_diagonal(sim_matrix, 1)
    
    return max_sim



def structural_similarity(smiles):

    smiles = np.array(smiles)

    all_mols = []

    for smile_id, smile in enumerate(smiles): 
        # print(smile_id)
        mol = Chem.MolFromSmiles(smile)
        all_mols.append(mol)


    # generate fingerprints 
    fpgen = AllChem.GetRDKitFPGenerator()

    fps = []
    
    for mol_id, x in enumerate(all_mols):
        try: 
            fps.append(fpgen.GetFingerprint(x))
        except: 
            print('invalid:', mol_id)

    # fps = [fpgen.GetFingerprint(x) for x in all_mols]

    sim_vals = []

    # use a triangular matrix: 
    for fp_id in range(len(fps)): 
        for fp_id_second_mol in range(len(fps)): 
            sim_vals.append(DataStructs.TanimotoSimilarity(fps[fp_id],fps[fp_id_second_mol]))
            
    # sim_matrix = squareform(np.array(sim_vals))
    # np.fill_diagonal(sim_matrix, 1)
    
    return sim_vals, fps, all_mols


def calculate_similarity(smiles, logBB): 

    logBB = np.array(logBB)
    delta_logBB_lst = []
    for mol_1 in range(len(logBB)): 
        for mol_2 in range(mol_1+1, len(logBB)): 
            delta_logBB_lst.append(np.abs(logBB[mol_1] - logBB[mol_2]))
    sim_matrix, sim_vals, fps = structural_similarity(smiles)

    return sim_vals, delta_logBB_lst


