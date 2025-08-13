# This function is used to evaluate the physical
# properties of the generated molecules.
# As a result, this function screens out all unsuitable molecules
# (not valid, not new, and with unsuitable physics and SA properties)
# in the final dataframe !df!
# or u can check variable 'total_with_sa3' - it is a sum of interesting molecules.
import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append('~')
sys.path.append(str(import_path))
sys.path.append(str(import_path)+'/../')
sys.path.append(str(import_path)+'/../classifier')
sys.path.append(str(import_path)+'/../../')
sys.path.append(str(import_path)+'/../../../../')
import statistics
from pipeline.classifier import create_sdf_file,Classifier
import pandas as pd
from rdkit.Contrib.SA_Score import sascorer
from rdkit import Chem,DataStructs
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw


path_to_generated_mols_csv = r'pipeline\coformers\mols_NC(=O)c1cnccn1.csv'
generated_mols = pd.read_csv(path_to_generated_mols_csv)['0']


drug = 'NC(=O)c1cnccn1'
classification = Classifier()
df = classification.clf_results(drug, generated_mols)

sa = []
for mol in df.iloc[:,1].tolist():
    sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))
total_with_sa3 = sum([i<=3 for i in sa])

list_smi = df.iloc[:,1].tolist()
fpgen = AllChem.GetRDKitFPGenerator()
df['mol'] = df.iloc[:,1].apply(Chem.MolFromSmiles)
fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in list_smi]

def check_diversity(mol):
    fp = fpgen.GetFingerprint(mol)
    scores = DataStructs.BulkTanimotoSimilarity(fp, fps)
    return statistics.mean(scores)

def check_chem_valid(smiles:List[str])->List[str]:
    """Check smiles for chemical validity and return only valid molecules.

    Args:
        smiles (List[str]): Molecules Smiles strings

    Returns:
        List[str]: Molecules Smiles strings
    """
    generated_coformers_clear = []
    for smiles_mol in smiles:
            if smiles_mol=='':
                continue
            if Chem.MolFromSmiles(str(smiles_mol)) is None:
                continue
            generated_coformers_clear.append(smiles_mol)
    generated_coformers_clear = [Chem.MolToSmiles(Chem.MolFromSmiles(str(i))) for i in generated_coformers_clear]
    return generated_coformers_clear

def check_novelty_mol_path(
        train_dataset_path: str,
        gen_data: list,
        train_col_name: str,
        gen_col_name: str,
        gen_len: int ):
    """Function for count how many new molecules generated compared with train data.


    :param train_dataset_path: Path to csv train dataset.
    :param gen_data: gen molecules.
    :param train_col_name: Name of column that consist a molecule strings.
    :param gen_col_name: Name of column that consist a molecule strings.
    :return:
    """
    train_d = pd.read_csv(train_dataset_path)[train_col_name]
    gen_d = pd.DataFrame(gen_data,columns=[gen_col_name])
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()/len(gen_d)
    total_len_gen = len(gen_d[gen_col_name])
    #gen_d = gen_d[gen_d['val_check']==1][gen_col_name]
    #len_train = len(train_d)
    len_gen = len(gen_d.drop_duplicates())
    novelty =( len(gen_d[gen_col_name].drop_duplicates())-gen_d[gen_col_name].drop_duplicates().isin(train_d).sum() )/ gen_len * 100
    print('Generated molecules consist of',novelty, '% unique new examples',
          '\t',
          f'duplicates: {duplicates}')
    return novelty,duplicates

def check_metrics(
        train_dataset_path: str,
        gen_data_path: str,
        train_col_name: str,
        gen_col_name: str) ->str:
    """Function for evaluate diversity and new,valid,duplicated molecules generated compared with train data.


    :param train_dataset_path: Path to csv train dataset.
    :param gen_data_path: Path to csv gen dataset.
    :param train_col_name: Name of column that consist a molecule strings.
    :param gen_col_name: Name of column that consist a molecule strings.
    :return:
    """
    train_d = pd.read_csv(train_dataset_path)[train_col_name]
    gen_d = pd.read_csv(gen_data_path)
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()
    total_len_gen = len(gen_d[gen_col_name])
    gen_d = gen_d[gen_d['val_check']==1][gen_col_name]
    #len_train = len(train_d)
    len_gen = len(gen_d)
    df['diversity'] = gen_d.mol.apply(check_diversity)
    mean_diversity= gen_d['diversity'].mean()

    print('Generated molecules consist of',(len_gen-train_d.isin(gen_d).sum())/len_gen*100, '% new examples',
          '\t',f'{len_gen/total_len_gen*100}% valid molecules generated','\t',
          f'duplicates, {duplicates}, mean diversity is {mean_diversity}')

def check_metrics_and_filter(
        train_dataset_path: str,
        gen_data_path: str,
        train_col_name: str,
        gen_col_name: str,
        filter_conditions: bool = True) -> pd.DataFrame:
    """
    Function for evaluating new/valid/duplicate molecules with optional filtering.
    
    Parameters:
        train_dataset_path: Path to training CSV
        gen_data_path: Path to generated CSV
        train_col_name: Name of molecule column in train dataset
        gen_col_name: Name of molecule column in generated dataset
        filter_conditions: Whether to apply additional filtering
    
    Returns:
        Filtered DataFrame and prints statistics
    """
    # Load data
    train_d = pd.read_csv(train_dataset_path)[train_col_name]
    gen_d = pd.read_csv(gen_data_path)
    
    # Remove duplicates in generated data
    duplicates = gen_d.duplicated(subset=gen_col_name, keep='first').sum()
    gen_d = gen_d.drop_duplicates(subset=gen_col_name, keep='first')
    
    # Keep only valid molecules
    gen_d = gen_d[gen_d['val_check'] == 1]
    
    # Find novel molecules (not present in training set)
    new_mols = ~gen_d[gen_col_name].isin(train_d)
    gen_d = gen_d[new_mols]
    
    # Calculate statistics
    total_initial = len(pd.read_csv(gen_data_path))
    total_after_processing = len(gen_d)
    
    print(f"Initially generated: {total_initial} molecules")
    print(f"After processing: {total_after_processing} molecules")
    print(f"Duplicates removed: {duplicates}")
    print(f"Percentage of novel molecules: {(len(gen_d)/total_initial)*100:.2f}%")
    
    # Apply additional filtering if requested
    if filter_conditions:
        filtered_df = gen_d[
            (gen_d['Unobstructed planes'] == 1) & 
            (gen_d['Orthogonal planes'] == 1) & 
            (gen_d['H-bonds bridging'] == 0)
        ]
        print(f"\nAfter additional filtering: {len(filtered_df)} molecules")
        print(f"Filter pass rate: {(len(filtered_df)/total_after_processing)*100:.2f}%")
        return filtered_df
    
    return gen_d


if __name__=='__main__':
    # Example usage
    check_metrics(train_dataset_path='',
                  gen_data_path='',
                  train_col_name='',
                  gen_col_name='')
    
    filtered_data = check_metrics_and_filter(
        train_dataset_path="",
        gen_data_path=".csv",
        train_col_name="",
        gen_col_name=""
    )

    # Save results
    filtered_data.to_csv("final_filtered.csv", index=False)

# df['diversity'] = df.mol.apply(check_diversity)
# mean_diversity= df['diversity'].mean()
# stop = 1
# print(df)
# sa= []
# for mol in df.iloc[:,1].tolist():
#     sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))
# df['sa_score']=sa
# df.to_csv(f'results/{drug}.csv')