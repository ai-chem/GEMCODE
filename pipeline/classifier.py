import os
import sys

import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
import pandas as pd
import numpy as np
import pickle as pi
import warnings
import random
import argparse
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from pathlib import Path
from GOLEM.scripts.utils import project_root
project_root = Path(__file__).parent.parent.parent
sys.path.append(os.getcwd())
warnings.filterwarnings('ignore')


class Classifier:

    def __init__(self):
        self.gbc_unobstructed = pi.load(open('classifier/checkpoints/gbc_Unobstructed.pkl', 'rb'))
        self.gbc_orthogonal_planes = pi.load(open('classifier/checkpoints/gbc_Orthogonal planes.pkl', 'rb'))
        self.gbc_h_bond_bridging = pi.load(open('classifier/checkpoints/gbc_H-bonds bridging.pkl', 'rb'))

        self.features_unobstructed = open('classifier/result_features/features_Unobstructed.txt','r').read().split('\n')
        self.features_orthogonal_planes = open('classifier/result_features/features_Orthogonal planes.txt','r').read().split('\n')
        self.features_h_bond_bridging = open('classifier/result_features/features_H-bonds bridging.txt','r').read().split('\n')

        self.min_max_scaler = pi.load(open('classifier/checkpoints/min_max_scaler.pkl', 'rb'))

        self.feature_num = 43

        self.desired_value = 1


    def get_drug_descriptors(self, drug):
        """Get coformers smiles

        Args:
            drug (str): smiles of drug

        Returns:
            table with drug descriptors
        """
        descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        num_descriptors = len(descriptor_names)
        descriptors_set = np.empty((0, num_descriptors), float)

        drug_obj = Chem.MolFromSmiles(drug)
        descriptors = np.array(get_descriptors.ComputeProperties(drug_obj)).reshape((-1,num_descriptors))
        descriptors_set = np.append(descriptors_set, descriptors, axis=0)
        drug_table = pd.DataFrame(descriptors_set, columns=descriptor_names)

        return drug_table

    def get_coformers_descriptors(self, generated_coformers):
        """Get coformers smiles

        Args:
            generated_coformers (list[str]): list of smiles

        Returns:
            table with coformers descriptors
        """
        descriptor_names = list(rdMolDescriptors.Properties.GetAvailableProperties())
        descriptor_coformer_names = [name + '.1' for name in descriptor_names]
        get_descriptors = rdMolDescriptors.Properties(descriptor_names)
        num_descriptors = len(descriptor_names)
        descriptors_set = np.empty((0, num_descriptors), float)

        if isinstance(generated_coformers, pd.Series):
            generated_coformers = generated_coformers.drop_duplicates().to_list()
        else:
            list(set(generated_coformers))
        generated_coformers_clear = []
        for smiles_mol in generated_coformers:
            if Chem.MolFromSmiles(str(smiles_mol)) is None:
                continue

            generated_coformers_clear.append(smiles_mol)
            gen_obj = Chem.MolFromSmiles(smiles_mol)
            descriptors = np.array(get_descriptors.ComputeProperties(gen_obj)).reshape((-1,num_descriptors))
            descriptors_set = np.append(descriptors_set, descriptors, axis=0)
            gen_table = pd.DataFrame(descriptors_set, columns=descriptor_coformer_names)

        return gen_table, generated_coformers_clear

    def create_clf_dataframe(self, drug, generated_coformers):
        """Create dataframe for classification

        Args:
            drug (str): smiles of drug
            generated_coformers (list[str]): list of smiles

        Returns:
            datafame of drug and coformers with descriptors
        """

        drug_table = self.get_drug_descriptors(drug)
        gen_table, generated_coformers_clear = self.get_coformers_descriptors(generated_coformers)

        clf_data = drug_table.merge(gen_table, how='cross')

        list_of_params = clf_data.columns.tolist()

        for feat_idx in range(self.feature_num):
            clf_data[list_of_params[feat_idx] + '_sum'] = \
                clf_data.iloc[:, feat_idx] + clf_data.iloc[:, feat_idx + self.feature_num]
            clf_data[list_of_params[feat_idx] + '_mean'] = \
                (clf_data.iloc[:, feat_idx] + clf_data.iloc[:, feat_idx + self.feature_num]) / 2

        clf_data_scaled = pd.DataFrame(self.min_max_scaler.transform(clf_data), columns=clf_data.columns)

        return clf_data_scaled, generated_coformers_clear

    def clf_data(self,generated_coformers, drug: str = 'Cn1c(=O)c2[nH]cnc2n(C)c1=O', part=100):
        """Classifier physic performance of data

        Args:
            drug (str): smiles of drug
            generated_coformers (list[str]): list of smiles

        Returns:
            dataframe: generated cocrystals with classifier score
        """
        generated_coformers = [i for i in generated_coformers if i!='']
        part_df = len(generated_coformers)//part
        df_total = pd.DataFrame(columns=['unobstructed','orthogonal_planes','h_bond_bridging', 0])
        for i in range(part):
            try:
                clf_data, generated_coformers_clear = self.create_clf_dataframe(drug,
                                                                                generated_coformers[i*part_df:(i+1)*part_df])
            except:
                continue
            df_total_temp = pd.DataFrame(data=generated_coformers_clear)
            #col = df_total.columns
            clf_data_unobstructed = pd.DataFrame(clf_data[self.features_unobstructed])
            clf_prediction_unobstructed = self.gbc_unobstructed.predict(clf_data_unobstructed)

            clf_data_orthogonal_planes = pd.DataFrame(clf_data[self.features_orthogonal_planes])
            clf_prediction_orthogonal_planes = (
                        self.gbc_orthogonal_planes.predict_proba(clf_data_orthogonal_planes)[:, 1] >= 0.332).astype(int)

            clf_data_h_bond_bridging = pd.DataFrame(clf_data[self.features_h_bond_bridging])
            clf_prediction_h_bond_bridging = self.gbc_h_bond_bridging.predict(clf_data_h_bond_bridging)

            df_total_temp['unobstructed'], df_total_temp['orthogonal_planes'], df_total_temp['h_bond_bridging'] = list(
                clf_prediction_unobstructed), list(clf_prediction_orthogonal_planes), list(clf_prediction_h_bond_bridging)
            df_total = pd.concat((df_total,df_total_temp))
        return df_total


    def clf_results(self, drug, generated_coformers, properties):
        """Classifier loss

        Args:
            drug (str): smiles of drug
            generated_coformers (list[str]): list of smiles

        Returns:
            dataframe: generated cocrystals with classifier score
        """

        clf_data, generated_coformers_clear = self.create_clf_dataframe(drug, generated_coformers)
        df_total = pd.DataFrame(data=generated_coformers_clear)


        clf_data_unobstructed = pd.DataFrame(clf_data[self.features_unobstructed])
        clf_prediction_unobstructed = self.gbc_unobstructed.predict(clf_data_unobstructed)

        clf_data_orthogonal_planes = pd.DataFrame(clf_data[self.features_orthogonal_planes])
        clf_prediction_orthogonal_planes = (self.gbc_orthogonal_planes.predict_proba(clf_data_orthogonal_planes)[:,1] >= 0.332).astype(int)

        clf_data_h_bond_bridging = pd.DataFrame(clf_data[self.features_h_bond_bridging])
        clf_prediction_h_bond_bridging = self.gbc_h_bond_bridging.predict(clf_data_h_bond_bridging)

        df_clf_results = pd.DataFrame(list(zip([drug] * len(generated_coformers_clear), generated_coformers_clear)),
                                      columns = ['drug', 'generated_coformers'])

        df_total['unobstructed'] = list(clf_prediction_unobstructed)

        df_clf_results['unobstructed'] = clf_prediction_unobstructed
        df_clf_results['orthogonal_planes'] = clf_prediction_orthogonal_planes
        df_clf_results['h_bond_bridging'] = clf_prediction_h_bond_bridging

        prop_values = {'unobstructed': 1, 'orthogonal_planes': 1, 'h_bond_bridging': 0}
        for prop in properties:
            df_clf_results = df_clf_results[df_clf_results[prop] == prop_values[prop]]
        return df_clf_results

def create_sdf_file(molecule, file_name):
    mol = Chem.MolFromSmiles(molecule)
    writer = Chem.SDWriter(file_name)
    writer.write(mol)
    writer.close()

def main(drug, model_path):

    gan_mol = pi.load(open(model_path, 'rb'))
    smiles_list = gan_mol.generate_n(1000)

    classification = Classifier()
    df = classification.clf_results(drug, smiles_list)

    create_sdf_file(drug, 'coformers/drug.sdf')

    for index, molecule in enumerate(df['generated_coformers']):
        create_sdf_file(molecule, './coformers/coformer_{}.sdf'.format(index))

    num_list = random.sample(range(1000), len(df))

    cc_table = ''

    for i in range(len(df['generated_coformers'])):
        cc_table += 'drug.sdf\tcoformer_{0}.sdf\t{1}\n'.format(i, num_list[i])

    with open('coformers/cc_table.csv', 'w') as f:
        f.write(cc_table)


def parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-drug', type=str,default='CC(=O)N1N=C(C2=CC=CC=C2F)CC1C1=CC=CC=C1Cl', help='')
    parser.add_argument('-model_path', type=str, default='generative_models\GAN\CCDC_fine-tuning\checkpoints\gan_mol_ccdc_256_1e-3_1k.pkl', help='')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parameter()
    main(args.drug, args.model_path)

