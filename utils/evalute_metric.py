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
df['diversity'] = df.mol.apply(check_diversity)
mean_diversity= df['diversity'].mean()
stop = 1
print(df)
sa= []
for mol in df.iloc[:,1].tolist():
    sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))
df['sa_score']=sa
df.to_csv(f'results/{drug}.csv')