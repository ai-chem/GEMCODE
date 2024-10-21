import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(import_path) + '/../')
sys.path.append(str(import_path) + '/../TVAE/generate')
sys.path.append(os.getcwd())

import old_module
sys.modules['allennlp'] = sys.modules['old_module']
sys.modules['allennlp.modules'] = sys.modules['old_module']
sys.modules['allennlp.modules.feedforward'] = sys.modules['old_module']
sys.modules['allennlp.modules.seq2seq_encoders'] = sys.modules['old_module']
sys.modules['allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper'] = sys.modules['old_module']

import argparse
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from openbabel import openbabel
openbabel.obErrorLog.StopLogging()
import pickle as pi
import pandas as pd
import random
from predict import main
from pipeline.classifier import Classifier
from pipeline.scripts.generate_vae import generator
from GOLEM.cocrystal_processing import run_experiment, molecule_search_setup, get_final_dataset

classification = Classifier()

# define a function to translate molecules into sdf files
def create_sdf_file(molecule, file_name):
    mol = Chem.MolFromSmiles(molecule)
    writer = Chem.SDWriter(file_name)
    writer.write(mol)
    writer.close()

def get_canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True)

def get_mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles)

def parameter():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='GAN', choices=['GAN', 'TVAE', 'TCVAE'])
    parser.add_argument('--drug', type=str, default='CN1C2=C(C(=O)N(C1=O)C)NC=N2')
    parser.add_argument('--num_molecules', type=int, default=100)
    parser.add_argument('--optimization', type=bool, default=False, choices=[True, False])
    parser.add_argument('--properties', type=str, nargs='+', default=['unobstructed', 'orthogonal_planes', 'h_bond_bridging'])
    
    return parser.parse_args()

if __name__ == "__main__":
    args = vars(parameter())
    generator_type = args['model']
    drug = args['drug']
    num_molecules_to_generate = args['num_molecules']
    optimization = args['optimization']
    properties = args['properties']

    # load the final model and generate smiles molecules
    conditions = [1,1,0]
    CONDITIONS = None if generator_type == "GAN" or generator_type == "TVAE" else conditions
    if generator_type != "GAN":
        df = generator(n_samples=num_molecules_to_generate,
                    path_to_save='.',
                    cuda=False,
                    save=False,
                    spec_conds = CONDITIONS, # chosee interested physycs conditions/None to initialize VAE generative model
                    weights_path=r'generative_models\TVAE\generate\for_generation') # preferably putt weights from google drive to this folder
        smiles_list = list(df[0])
        
    else:
        print('generating samples...')
        gan_mol = pi.load(open('generative_models/GAN/CCDC_fine-tuning/checkpoints/gan_mol_ccdc_256_1e-3_1k.pkl', 'rb'))
        smiles_list = gan_mol.generate_n(num_molecules_to_generate)
    
    # select the coformer molecules that form a co-crystal with the best mechanical properties for co-crystal tableting 
    df = classification.clf_results(drug, smiles_list, properties)

    # create the directory 
    path_coformer = os.path.join("pipeline/", "coformers") 
    if os.path.isdir(path_coformer) == False:
        os.mkdir(path_coformer)     

    # run evolutionary optimization
    if optimization:
        init_path = f'pipeline/coformers/mols_{drug}.csv'
        df.to_csv(init_path)

        col_name = 'generated_coformers'
        results_folder = 'pipeline/results_golem'
        exp_ids = []
        exp_id = run_experiment(molecule_search_setup,
                                initial_data_path=init_path,
                                col_name=col_name,
                                max_heavy_atoms=50,
                                trial_timeout=60,
                                trial_iterations=5,
                                pop_size=20,
                                visualize=True,
                                num_trials=2,
                                drug=drug,
                                result_dir=results_folder
                                )
        exp_ids.append(exp_id)
        exp_results_folder = os.path.join(results_folder, exp_id)
        df = get_final_dataset(exp_results_folder, drug)

    # create sdf files
    create_sdf_file(drug, 'pipeline/coformers/drug.sdf')
    for index, molecule in enumerate(df['generated_coformers']):
        create_sdf_file(molecule, 'pipeline/coformers/coformer_{}.sdf'.format(index))
    
    # generate a list of random values for ccgnet
    num_list = random.sample(range(len(df)), len(df))
    
    # create a csv file to run ccgnet
    cc_table = ''

    # create a csv file to run ccgnet
    for i in range(len(df['generated_coformers'])):
        cc_table += 'drug.sdf\tcoformer_{0}.sdf\t{1}\n'.format(i, num_list[i])

    with open('pipeline/coformers/cc_table.csv', 'w') as f:
        f.write(cc_table)
    
    # rank the coformers by probability of co-crystallisation and save the result to a file 'ccgnet_output.xlsx'
    main('pipeline/coformers/cc_table.csv', 'pipeline/coformers', fmt='sdf', model_type='cc', xlsx_name='pipeline/result/ccgnet_output.xlsx')

    # load the CCGNet output Excel file into a df
    df_ccgnet = pd.read_excel('pipeline/result/ccgnet_output.xlsx') 

    # apply the functions
    df['generated_coformers'] = df['generated_coformers'].apply(get_canonical_smiles)
    df_ccgnet['generated_coformers'] = df_ccgnet['SMILES.1'].apply(get_canonical_smiles)

    # merge the two dfs on the 'generated_coformers' column
    merged_df = pd.merge(df_ccgnet[['generated_coformers', 'Score']], df[['drug', 'generated_coformers', 'unobstructed', 'orthogonal_planes', 'h_bond_bridging']], 
                     left_on='generated_coformers', 
                     right_on='generated_coformers', 
                     how='left')
    
    # reorder columns and drop duplicates
    merged_df = merged_df[['drug', 'generated_coformers', 'unobstructed', 'orthogonal_planes', 'h_bond_bridging', 'Score']]
    merged_df = merged_df.drop_duplicates(subset=['generated_coformers'])
    
    # insert images
    merged_df.insert(1, "drug_img", merged_df["drug"].apply(get_mol_from_smiles))
    merged_df.insert(3, "coformer_img", merged_df["generated_coformers"].apply(get_mol_from_smiles))

    # save the result to a file 'cocrystals.xlsx'
    PandasTools.SaveXlsxFromFrame(merged_df, 'pipeline/result/cocrystals.xlsx', molCol=['drug_img', 'coformer_img'], size=(100, 100))
