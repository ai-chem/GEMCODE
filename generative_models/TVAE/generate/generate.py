import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(import_path)
sys.path.append(str(import_path))
sys.path.append(str(import_path)+'/../')
sys.path.append(str(import_path)+'/../classifier')
sys.path.append(str(import_path)+'/../../')
sys.path.append(str(import_path)+'/../../../../')

from inference import generate
from Process import *
import argparse
from Models import get_model
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from pipeline.classifier import Classifier

def generator(n_samples=1000,
              path_to_save:str='_gen_mols',
              spec_conds=[1,1,0],
              save:bool=True,
              cuda:bool=False,
              weights_path:str = "generative_models\TVAE\generate//for_generation",
              drug:str = 'CN1C2=C(C(=O)N(C1=O)C)NC=N2'
              ):
    '''
    The generator function generates the specified number of molecules.
        n_samples - number of molecules to be generated.
        path_to_save - suffix to the file path to save the molecules.
    It is necessary to give a name to the file for generation.
        save - whether to save the generated molecules? True/False
        spec_conds - None for random assignment of physical properties/
    list for specifying molecules of interest. Example: [1,1,0].
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=str, default=weights_path)
    #parser.add_argument('-load_traindata', type=str, default="data/moses/prop_temp.csv")
    parser.add_argument('-load_toklendata', type=str, default='toklen_list.csv')
    parser.add_argument('-k', type=int, default=4)
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-max_strlen', type=int, default=100)  # max 80
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)

    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-cond_dim', type=int, default=3)
    parser.add_argument('-latent_dim', type=int, default=128)

    parser.add_argument('-epochs', type=int, default=30)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-lr_beta1', type=int, default=0.9)
    parser.add_argument('-lr_beta2', type=int, default=0.98)

    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-cuda', type=str, default=cuda)
    parser.add_argument('-floyd', action='store_true')
    data_path = import_path + '/../../GAN'
    parser.add_argument('-cond_test_path', type=str,
                        default=f'{data_path}/CCDC_fine-tuning/data/physic/conditions/database_CCDC_cond_test.csv')
    parser.add_argument('-cond_train_path', type=str,
                        default=f'{data_path}/CCDC_fine-tuning/data/physic/conditions/database_CCDC_cond_train.csv')
    parser.add_argument('-train_data_csv', type=str,
                        default=f'{data_path}/CCDC_fine-tuning/data/physic/molecules/database_CCDC_train.csv')  # Need to check novelty of generated mols from train data
    opt = parser.parse_args()

    opt.device = 'cuda' if opt.cuda is True else 'cpu'
    opt.path_script = import_path
    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    clssr = Classifier()
    opt.classifire = clssr
    df = generate(opt=opt,
             model=model,
             SRC=SRC,
             TRG=TRG,
             n_samples=n_samples,
             spec_conds=spec_conds,
             save=save,
             shift_path=path_to_save,
             drug=drug)
    return df



if __name__ == '__main__':
    generator()