from classifier import Classifier
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def count_physic_perf( path_to_data:str = None,
                      path_to_save:str = None,
                      drug: str = 'Cn1c(=O)c2[nH]cnc2n(C)c1=O',
                       data_column:str = '0')-> None:
    """

    :param path_to_data: Path to csv data with molecules.
    :param path_to_save: Path to save csv data with molecules.
    :param drug: Interested drug molecule.
    :return: None. Saved data format "molecules,'unobstructed', 'orthogonal_planes', 'h_bond_bridging'"
    """
    classification = Classifier()
    smiles_list = list(pd.read_csv(path_to_data)[data_column])
    df = classification.clf_data( smiles_list,drug=drug)
    df.to_csv(path_to_save,index=False)


if __name__ == '__main__':

    count_physic_perf(path_to_data='',
                      path_to_save='',
                      data_column='')