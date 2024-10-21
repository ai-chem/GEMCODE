import sys
import os
import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)
sys.path.append(str(import_path))
sys.path.append(str(import_path)+'/../')
sys.path.append(str(import_path)+'/../classifier')
sys.path.append(str(import_path)+'/../../')
sys.path.append(str(import_path)+'/../../../../')

from generative_models.TVAE.inference import generate
from generative_models.TVAE.Process import *
from generative_models.TVAE.Models import get_model
from pipeline.classifier import Classifier

from argparse import Namespace
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

def generator(n_samples=10,
              path_to_save:str='_generated_reacts',
              spec_conds=None,
              save:bool=True,
              cuda:bool=False,
              weights_path:str = "generative_models\TVAE\generate//for_generation"):

    data_path = import_path + '/../../GAN'


    opt = Namespace(classifire=Classifier(), 
                    cond_dim=3, 
                    cond_test_path=f'{data_path}/CCDC_fine-tuning/data/physic/conditions/database_CCDC_cond_test.csv',
                    cond_train_path=f'{data_path}/CCDC_fine-tuning/data/physic/conditions/database_CCDC_cond_train.csv', 
                    cuda=cuda, 
                    d_model=512, 
                    device='cuda' if cuda is True else 'cpu', 
                    dropout=0.1, 
                    epochs=30, 
                    floyd=False, 
                    heads=8, 
                    k=4, 
                    lang_format='SMILES', 
                    latent_dim=128, 
                    load_toklendata='toklen_list.csv', 
                    load_weights=weights_path, 
                    lr=0.0001, lr_beta1=0.9, 
                    lr_beta2=0.98, 
                    max_strlen=100, 
                    n_layers=6, 
                    path_script=import_path, 
                    print_model=False, 
                    train_data_csv=f'{data_path}/CCDC_fine-tuning/data/physic/molecules/database_CCDC_train.csv', 
                    use_cond2dec=False, 
                    use_cond2lat=True)

    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    df = generate(opt=opt,
             model=model,
             SRC=SRC,
             TRG=TRG,
             n_samples=n_samples,
             spec_conds=spec_conds,
             save=save,
             shift_path=path_to_save)
    
    return df

if __name__ == '__main__':
    generator()