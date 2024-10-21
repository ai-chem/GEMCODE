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
import time
from Process import *
import argparse
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, QED, rdDepictor, AllChem, Draw
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import joblib
import numpy as np
from rand_gen import rand_gen_from_data_distribution, tokenlen_gen_from_data_distribution
from dataDistibutionCheck import checkdata
from tqdm import tqdm
from utils.check_novelty import check_novelty_mol_path
import warnings
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
from rdkit.Contrib.SA_Score import sascorer




def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0

def gen_mol(cond, model, opt, SRC, TRG, toklen, z):
    model.eval()

    robustScaler = joblib.load(opt.load_weights + '/scaler.pkl')
    if opt.conds == 'm':
        cond = cond.reshape(1, -1)
    elif opt.conds == 's':
        cond = cond.reshape(1, -1)
    elif opt.conds == 'l':
        cond = cond.reshape(1, -1)
    else:
        cond = np.array(cond.split(',')[:-1]).reshape(1, -1)

    cond = robustScaler.transform(cond)
    cond = Variable(torch.Tensor(cond))
    
    sentence = beam_search(cond, model, SRC, TRG, toklen, opt, z)
    return sentence


def gen_mol_val_vae(cond, model, opt, SRC, TRG, toklen, z):
    model.eval()
    cond = Variable(torch.Tensor(cond))
    sentence = beam_search(cond, model, SRC, TRG, toklen, opt, z)
    return sentence

def gen_mol_val(cond, model, opt, SRC, TRG, toklen, z):
    model.eval()
    cond = Variable(torch.Tensor(cond))
    sentence = beam_search(cond, model, SRC, TRG, toklen, opt, z)
    return sentence

def validate_vae(opt, model, SRC, TRG,n_samples=100,spec_conds=None,save=False,shift_path=''):
    warnings.filterwarnings('ignore')
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    #print("\nGenerating molecules for MOSES benchmarking...")
    n_samples = n_samples
    #nBins = [100, 100, 100]

    #data = pd.read_csv(opt.cond_train_path)
    toklen_data = pd.read_csv("{opt.path_script}/weights/toklen_list.csv")
    # if spec_conds is None:
    #     conds = np.array([[np.random.randint(2),np.random.randint(2),np.random.randint(2)] for i in range(n_samples)])# for physics performances
    # else:
    #     conds = np.array([list(spec_conds) for _ in range(n_samples)])
    #conds = np.array([[np.random.randint(2),np.random.randint(2),np.random.randint(2)] for i in range(n_samples)])# for physics performances
    #conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)

    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)
    opt.conds = 'm'
    start = time.time()
    for idx in tqdm(range(n_samples)):
        toklen = int(toklen_data[idx]) + 3  # +3 due to cond2enc
        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

        molecules.append(molecule_tmp)
    df_init = pd.DataFrame(np.concatenate((conds, np.array(molecules).reshape(len(molecules), 1)), axis=1),columns= ['unobstructed', 'orthogonal_planes', 'h_bond_bridging', 0])

    clf_df = opt.classifire.clf_data(generated_coformers=molecules, part=1)
    sa = []
    for mol in clf_df[0].tolist():
        sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))

    if save:
        df_to_save = clf_df.copy(deep=True)
        df_to_save['sa'] = sa
        df_to_save.drop_duplicates().to_csv(f'{opt.path_script}/valid_mols_wo_dupls{shift_path}.csv')
        df_init.to_csv(f'{opt.path_script}/generated_mols{shift_path}.csv')

    sa_total =pd.DataFrame(sa,columns=['mol'])
    df_total = df_init.merge(clf_df, right_on=0, left_on=0, how='right')
    valid = sum([Chem.MolFromSmiles(i) is not None for i in df_init[0]]) / n_samples


    novelty, duplicates = check_novelty_mol_path(train_dataset_path=opt.train_data_csv, gen_data=df_total[0].to_list(),
                                                 train_col_name='0.1', gen_col_name='mol',gen_len = len(molecules))
    df_total = df_total.drop_duplicates()
    df_total = df_total[~df_total[0].isin(pd.read_csv(opt.train_data_csv)['0.1'])]

    init_cond = np.int_(df_total.iloc[:,:3].to_numpy())
    generated_cond = df_total.iloc[:,4:].to_numpy()


    #df_cond_comp - DF, that consist of molecules, with matched conditions performances
    df_cond_comp = df_total[(df_total['unobstructed_x'].astype(int)==df_total[ 'unobstructed_y'])&(df_total['orthogonal_planes_x'].astype(int)==df_total['orthogonal_planes_y'])&(df_total['h_bond_bridging_x'].astype(int)==df_total['h_bond_bridging_y'])]
    correctness = [all(i[0]==i[1]) for i in zip(init_cond,generated_cond)]#without duplicates and only valids
    sa_unique = []
    for mol in df_cond_comp[0].tolist():
        sa_unique.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))

    sa_cond_compared = pd.DataFrame(sa_unique, columns=['mol'])
    sa_comp_3 = (sa_cond_compared['mol']<=3).sum()
    of_ineterested_cond = sa_comp_3/len(df_init) #How many molecules have SA<=3 of mol-s,
                                                            # that have ineterested physics performance
    of_total_generated = sa_comp_3/n_samples
    total_sa_3 = (sa_total['mol'] <= 3).sum()
    if correctness == []:
        cond_score = 0
    else:
        cond_score = sum(correctness)/len(df_init)

    ###
    list_smi = df_cond_comp[0].tolist()
    fpgen = AllChem.GetRDKitFPGenerator()
    df_cond_comp['mol'] = df_cond_comp[0].apply(Chem.MolFromSmiles)
    fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in list_smi]

    def check_diversity(mol):
        fp = fpgen.GetFingerprint(mol)
        scores = DataStructs.BulkTanimotoSimilarity(fp, fps)
        return statistics.mean(scores)

    df_cond_comp['diversity'] = df_cond_comp.mol.apply(check_diversity)
    ###

    print('Condition correct score (how many conditions matched):', cond_score, 'Total valid mol:', valid,
          'Molecules of spec cond with sa<=3:', of_ineterested_cond, ';Matched conds', len(df_cond_comp),
          'Spec cond:', spec_conds, 'Mean_deversity:', df_cond_comp['diversity'].mean(), ';Sum intersting molecules:',
          sa_comp_3)
    return cond_score, valid, novelty, duplicates, of_ineterested_cond, df_cond_comp['diversity'].mean()


def generate(opt, model, SRC, TRG,n_samples=100,spec_conds=None,save=False,shift_path='',drug='CN1C2=C(C(=O)N(C1=O)C)NC=N2'):
    warnings.filterwarnings('ignore')
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    #print("\nGenerating molecules for MOSES benchmarking...")
    n_samples = n_samples
    #nBins = [100, 100, 100]

    #data = pd.read_csv(opt.cond_train_path)
    #toklen_data = pd.read_csv("weights/toklen_list.csv")
    toklen_data = pd.read_csv(f'{opt.load_weights}/toklen_list.csv')

    if spec_conds is None:
        conds = np.array([[np.random.randint(2),np.random.randint(2),np.random.randint(2)] for i in range(n_samples)])# for physics performances
    else:
        conds = np.array([list(spec_conds) for _ in range(n_samples)])

    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)
    opt.conds = 'm'
    start = time.time()
    for idx in tqdm(range(n_samples)):
        toklen = int(toklen_data[idx]) + 3  # +3 due to cond2enc
        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

        molecules.append(molecule_tmp)
    df_init = pd.DataFrame(np.concatenate((conds, np.array(molecules).reshape(len(molecules), 1)), axis=1),columns= ['unobstructed', 'orthogonal_planes', 'h_bond_bridging', 0])

    clf_df = opt.classifire.clf_data(generated_coformers=molecules, part=1,drug=drug)
    sa = []
    for mol in clf_df[0].tolist():
        sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))
    df_to_save = clf_df.copy(deep=True)
    df_to_save['sa'] = sa
    if save:
        df_to_save.drop_duplicates().to_csv(f'{opt.path_script}/valid_mols_wo_dupls{shift_path}.csv')
        df_init.to_csv(f'{opt.path_script}/generated_mols{shift_path}.csv')
    return df_to_save.drop_duplicates()


def validate(opt, model, SRC, TRG,n_samples=100,spec_conds=None,save=False,shift_path='',is_molecules=True):
    warnings.filterwarnings('ignore')
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    #print("\nGenerating molecules for MOSES benchmarking...")
    n_samples = n_samples
    #nBins = [100, 100, 100]

    #data = pd.read_csv(opt.cond_train_path)
    toklen_data = pd.read_csv(f"{opt.path_script}/weights/toklen_list.csv")
    if spec_conds is None:
        conds = np.array([[np.random.randint(2),np.random.randint(2),np.random.randint(2)] for i in range(n_samples)])# for physics performances
    else:
        conds = np.array([list(spec_conds) for _ in range(n_samples)])
    #conds = np.array([[np.random.randint(2),np.random.randint(2),np.random.randint(2)] for i in range(n_samples)])# for physics performances
    #conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)

    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)
    opt.conds = 'm'
    start = time.time()
    for idx in tqdm(range(n_samples)):
        toklen = int(toklen_data[idx]) + 3  # +3 due to cond2enc
        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

        molecules.append(molecule_tmp)
    df_init = pd.DataFrame(np.concatenate((conds, np.array(molecules).reshape(len(molecules), 1)), axis=1),columns= ['unobstructed', 'orthogonal_planes', 'h_bond_bridging', 0])

    clf_df = opt.classifire.clf_data(generated_coformers=molecules, part=1)
    sa = []
    for mol in clf_df[0].tolist():
        sa.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))

    if save:
        df_to_save = clf_df.copy(deep=True)
        df_to_save['sa'] = sa
        df_to_save.drop_duplicates().to_csv(f'{opt.path_script}/valid_mols_wo_dupls{shift_path}.csv')
        df_init.to_csv(f'{opt.path_script}/generated_mols{shift_path}.csv')

    sa_total =pd.DataFrame(sa,columns=['mol'])
    df_total = df_init.merge(clf_df, right_on=0, left_on=0, how='right')
    if is_molecules:
        valid = sum([Chem.MolFromSmiles(i) is not None for i in df_init[0]]) / n_samples
    else:
        reacts = []
        for i in df_init[0]:
            try:
                react_temp = AllChem.ReactionFromSmarts(i)
                reacts.append(True)
            except:
                reacts.append(False)
        valid = sum(reacts)/ n_samples


    novelty, duplicates = check_novelty_mol_path(train_dataset_path=opt.train_data_csv, gen_data=df_total[0].to_list(),
                                                 train_col_name='0.1', gen_col_name='mol',gen_len = len(molecules))
    df_total = df_total.drop_duplicates()
    df_total = df_total[~df_total[0].isin(pd.read_csv(opt.train_data_csv)['0.1'])]

    init_cond = np.int_(df_total.iloc[:,:3].to_numpy())
    generated_cond = df_total.iloc[:,4:].to_numpy()


    #df_cond_comp - DF, that consist of molecules, with matched conditions performances
    df_cond_comp = df_total[(df_total['unobstructed_x'].astype(int)==df_total[ 'unobstructed_y'])&(df_total['orthogonal_planes_x'].astype(int)==df_total['orthogonal_planes_y'])&(df_total['h_bond_bridging_x'].astype(int)==df_total['h_bond_bridging_y'])]
    correctness = [all(i[0]==i[1]) for i in zip(init_cond,generated_cond)]#without duplicates and only valids
    sa_unique = []
    for mol in df_cond_comp[0].tolist():
        sa_unique.append(sascorer.calculateScore(Chem.MolFromSmiles(mol)))

    sa_cond_compared = pd.DataFrame(sa_unique, columns=['mol'])
    sa_comp_3 = (sa_cond_compared['mol']<=3).sum()
    of_ineterested_cond = sa_comp_3/len(df_init) #How many molecules have SA<=3 of mol-s,
                                                            # that have ineterested physics performance
    of_total_generated = sa_comp_3/n_samples
    total_sa_3 = (sa_total['mol'] <= 3).sum()
    if correctness == []:
        cond_score = 0
    else:
        cond_score = sum(correctness)/len(df_init)

    ###
    list_smi = df_cond_comp[0].tolist()
    fpgen = AllChem.GetRDKitFPGenerator()
    df_cond_comp['mol'] = df_cond_comp[0].apply(Chem.MolFromSmiles)
    fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(x)) for x in list_smi]

    def check_diversity(mol):
        fp = fpgen.GetFingerprint(mol)
        scores = DataStructs.BulkTanimotoSimilarity(fp, fps)
        return statistics.mean(scores)

    df_cond_comp['diversity'] = df_cond_comp.mol.apply(check_diversity)
    ###

    print('Condition correct score (how many conditions matched):', cond_score, 'Total valid mol:', valid,
          'Molecules of spec cond with sa<=3:', of_ineterested_cond, ';Matched conds', len(df_cond_comp),
          'Spec cond:', spec_conds, 'Mean_deversity:', df_cond_comp['diversity'].mean(), ';Sum intersting molecules:',
          sa_comp_3)
    return cond_score, valid, novelty, duplicates, of_ineterested_cond, df_cond_comp['diversity'].mean()
        # if (idx+1) % (n_samples)//10 == 0:
        #     print("*   {}m: {} / {}".format((time.time() - start)//60, idx+1, n_samples))
        #
        # if (idx+1) % (n_samples)//10 == 0:
        #     np_conds_trg, np_conds_rdkit = np.array(conds_trg), np.array(conds_rdkit)
        #     gen_list = pd.DataFrame(
        #         {"mol": molecules, "val_check": val_check, "trg(logP)": np_conds_trg[:, 0], "trg(tPSA)": np_conds_trg[:, 1], "trg(QED)": np_conds_trg[:, 2], "rdkit(logP)": np_conds_rdkit[:, 0], "rdkit(tPSA)": np_conds_rdkit[:, 1], "rdkit(QED)": np_conds_rdkit[:, 2], "toklen": toklen_check, "toklen_gen": toklen_gen})
        #     gen_list.to_csv('moses_bench2_lat={}_epo={}_k={}_{}.csv'.format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")), index=True)


def inference_phys(opt, model, SRC, TRG,n_samples=100,spec_conds = None):
    warnings.filterwarnings('ignore')
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    #print("\nGenerating molecules for MOSES benchmarking...")
    n_samples = n_samples
    nBins = [100, 100, 100]

    data = pd.read_csv(opt.cond_train_path)
    toklen_data = pd.read_csv(f"{opt.path_script}/weights/toklen_list.csv")
    if spec_conds is None:
        conds = np.array([[np.random.randint(2),np.random.randint(2),np.random.randint(2)] for i in range(n_samples)])# for physics performances
    else:
        conds = np.array([spec_conds] for _ in range(n_samples))

    #conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)

    toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)
    opt.conds = 'm'
    start = time.time()
    for idx in range(n_samples):
        toklen = int(toklen_data[idx]) + 3  # +3 due to cond2enc
        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
        molecule_tmp = gen_mol_val(conds[idx], model, opt, SRC, TRG, toklen, z)
        toklen_gen.append(molecule_tmp.count(' ')+1)
        molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

        molecules.append(molecule_tmp)
        #conds_trg.append(conds[idx])
        # toklen-3: due to cond dim
        #toklen_check.append(toklen-3)
        #m = Chem.MolFromSmiles(molecule_tmp)
        # Uncomment for NOT physics case
        # if m is None:
        #     val_check.append(0)
        #     conds_rdkit.append([None, None, None])
        # else:
        #     val_check.append(1)
        #     conds_rdkit.append(np.array([Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)]))
    df_init = pd.DataFrame(np.concatenate((conds, np.array(molecules).reshape(len(molecules), 1)), axis=1),columns= ['unobstructed', 'orthogonal_planes', 'h_bond_bridging', 0])

    clf_df = opt.classifire.clf_data(generated_coformers=molecules, part=1)
    df_total = df_init.merge(clf_df, right_on=0, left_on=0, how='right')
    init_cond = np.int_(df_total.iloc[:,:3].to_numpy())
    generated_cond = df_total.iloc[:,4:].to_numpy()
    valid = len(df_total)/len(df_init)
    correctness = [all(i[0]==i[1]) for i in zip(init_cond,generated_cond)]
    #f1 = [f1_score(list(i[0]),list(i[1]),zero_division=1.0) for i in zip(init_cond,generated_cond)]
    if correctness == []:
        cond_score = 0
    else:
        cond_score = sum(correctness)/len(correctness)

    check_novelty_mol_path(train_dataset_path= opt.train_data_csv,gen_data=df_total[0].to_list(),train_col_name='0.1',gen_col_name='mol')
    print('Condition correct score:',cond_score,'Total valid mol:',valid)
    return df_total
        # if (idx+1) % (n_samples)//10 == 0:
        #     print("*   {}m: {} / {}".format((time.time() - start)//60, idx+1, n_samples))
        #
        # if (idx+1) % (n_samples)//10 == 0:
        #     np_conds_trg, np_conds_rdkit = np.array(conds_trg), np.array(conds_rdkit)
        #     gen_list = pd.DataFrame(
        #         {"mol": molecules, "val_check": val_check, "trg(logP)": np_conds_trg[:, 0], "trg(tPSA)": np_conds_trg[:, 1], "trg(QED)": np_conds_trg[:, 2], "rdkit(logP)": np_conds_rdkit[:, 0], "rdkit(tPSA)": np_conds_rdkit[:, 1], "rdkit(QED)": np_conds_rdkit[:, 2], "toklen": toklen_check, "toklen_gen": toklen_gen})
        #     gen_list.to_csv('moses_bench2_lat={}_epo={}_k={}_{}.csv'.format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")), index=True)


def inference(opt, model, SRC, TRG):
    molecules, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    if opt.conds == 'm':
        print("\nGenerating molecules for MOSES benchmarking...")
        n_samples = 100
        nBins = [100, 100, 100]

        data = pd.read_csv(opt.load_traindata)
        toklen_data = pd.read_csv(opt.load_toklendata)

        conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)
        toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples)

        start = time.time()
        for idx in range(n_samples):
            toklen = int(toklen_data[idx]) + 3  # +3 due to cond2enc
            z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
            molecule_tmp = gen_mol(conds[idx], model, opt, SRC, TRG, toklen, z)
            toklen_gen.append(molecule_tmp.count(' ')+1)
            molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

            molecules.append(molecule_tmp)
            conds_trg.append(conds[idx])
            # toklen-3: due to cond dim
            toklen_check.append(toklen-3)
            m = Chem.MolFromSmiles(molecule_tmp)
            if m is None:
                val_check.append(0)
                conds_rdkit.append([None, None, None])
            else:
                val_check.append(1)
                conds_rdkit.append(np.array([Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)]))

            if (idx+1) % (n_samples)//10 == 0:
                print("*   {}m: {} / {}".format((time.time() - start)//60, idx+1, n_samples))

            if (idx+1) % (n_samples)//10 == 0:
                np_conds_trg, np_conds_rdkit = np.array(conds_trg), np.array(conds_rdkit)
                gen_list = pd.DataFrame(
                    {"mol": molecules, "val_check": val_check, "trg(logP)": np_conds_trg[:, 0], "trg(tPSA)": np_conds_trg[:, 1], "trg(QED)": np_conds_trg[:, 2], "rdkit(logP)": np_conds_rdkit[:, 0], "rdkit(tPSA)": np_conds_rdkit[:, 1], "rdkit(QED)": np_conds_rdkit[:, 2], "toklen": toklen_check, "toklen_gen": toklen_gen})
                gen_list.to_csv('moses_bench2_lat={}_epo={}_k={}_{}.csv'.format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")), index=True)

        print("Please check the file: 'moses_bench2_lat={}_epo={}_k={}_{}.csv'".format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")))


    elif opt.conds == 's':
        print("\nGenerating molecules for 10 condition sets...")
        n_samples = 10
        n_per_samples = 200
        nBins = [1000, 1000, 1000]

        data = pd.read_csv(opt.load_traindata)
        toklen_data = pd.read_csv(opt.load_toklendata)

        conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)
        toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max()-toklen_data.min()), size=n_samples*n_per_samples)

        print("conds:\n", conds)
        start = time.time()
        for idx in range(n_samples):
            for i in range(n_per_samples):
                toklen = int(toklen_data[idx*n_per_samples + i]) + 3  # +3 due to cond2enc
                z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
                molecule_tmp = gen_mol(conds[idx], model, opt, SRC, TRG, toklen, z)
                toklen_gen.append(molecule_tmp.count(" ") + 1)
                molecule_tmp = ''.join(molecule_tmp).replace(" ", "")

                molecules.append(molecule_tmp)
                conds_trg.append(conds[idx])

                toklen_check.append(toklen-3) # toklen -3: due to cond size
                m = Chem.MolFromSmiles(molecule_tmp)
                if m is None:
                    val_check.append(0)
                    conds_rdkit.append([None, None, None])
                else:
                    val_check.append(1)
                    conds_rdkit.append(np.array([Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)]))

                if (idx*n_per_samples+i+1) % 100 == 0:
                    print("*   {}m: {} / {}".format((time.time() - start)//60, idx*n_per_samples+i+1, n_samples*n_per_samples))

                if (idx*n_per_samples+i+1) % 200 == 0:
                    np_conds_trg, np_conds_rdkit = np.array(conds_trg), np.array(conds_rdkit)
                    gen_list = pd.DataFrame(
                        {"set_idx": idx, "mol": molecules, "val_check": val_check, "trg(logP)": np_conds_trg[:, 0], "trg(tPSA)": np_conds_trg[:, 1], "trg(QED)": np_conds_trg[:, 2], "rdkit(logP)": np_conds_rdkit[:, 0], "rdkit(tPSA)": np_conds_rdkit[:, 1], "rdkit(QED)": np_conds_rdkit[:, 2], "toklen": toklen_check, "toklen_gen": toklen_gen})
                    gen_list.to_csv('moses_bench2_10conds_lat={}_epo={}_k={}_{}.csv'.format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")), index=True)

        print("Please check the file: 'moses_bench2_10conds_lat={}_epo={}_k={}_{}.csv'".format(opt.latent_dim, opt.epochs, opt.k, time.strftime("%Y%m%d")))

    else:
        conds = opt.conds.split(';')
        toklen_data = pd.read_csv(opt.load_toklendata)
        toklen= int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max() - toklen_data.min()), size=1)) + 3  # +3 due to cond2enc

        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))

        for cond in conds:
            molecules.append(gen_mol(cond + ',', model, opt, SRC, TRG, toklen, z))
        toklen_gen = molecules[0].count(" ") + 1
        molecules = ''.join(molecules).replace(" ", "")
        m = Chem.MolFromSmiles(molecules)
        target_cond = conds[0].split(',')
        if m is None:
            #toklen-3: due to cond dim
            print("   --[Invalid]: {}".format(molecules))
            print("   --Target: logP={}, tPSA={}, QED={}, LatentToklen={}".format(target_cond[0], target_cond[1], target_cond[2], toklen-3))
        else:
            logP_v, tPSA_v, QED_v = Descriptors.MolLogP(m), Descriptors.TPSA(m), QED.qed(m)
            print("   --[Valid]: {}".format(molecules))
            print("   --Target: logP={}, tPSA={}, QED={}, LatentToklen={}".format(target_cond[0], target_cond[1], target_cond[2], toklen-3))
            print("   --From RDKit: logP={:,.4f}, tPSA={:,.4f}, QED={:,.4f}, GenToklen={}".format(logP_v, tPSA_v, QED_v, toklen_gen))

    return molecules


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=str, default=f"{opt.path_script}/old_weights")
    parser.add_argument('-load_traindata', type=str, default="data/moses/prop_temp.csv")
    parser.add_argument('-load_toklendata', type=str, default='toklen_list.csv')
    parser.add_argument('-k', type=int, default=4)
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-max_strlen', type=int, default=100) #max 80
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)

    parser.add_argument('-use_cond2dec', type=bool, default=False)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-cond_dim', type=int, default=3)
    parser.add_argument('-latent_dim', type=int, default=128)

    parser.add_argument('-epochs', type=int, default=1111111111111)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-lr_beta1', type=int, default=0.9)
    parser.add_argument('-lr_beta2', type=int, default=0.98)

    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    
    opt = parser.parse_args()

    opt.device = 'cuda' if opt.no_cuda is False else -1

    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.max_logP, opt.min_logP, opt.max_tPSA, opt.min_tPSA, opt.max_QED, opt.min_QED = checkdata(opt.load_traindata)

    while True:
        opt.conds =input("\nEnter logP, tPSA, QED to generate molecules (refer the pop-up data distribution)\
        \n* logP: {:.2f} ~ {:.2f}; tPSA: {:.2f} ~ {:.2f}; QED: {:.2f} ~ {:.2f} is recommended.\
        \n* Typing sample: 2.2, 85.3, 0.8\n* Enter the properties (Or type m: MOSES benchmark, s: 10-Condition set test, q: quit):".format(opt.min_logP, opt.max_logP, opt.min_tPSA, opt.max_tPSA, opt.min_QED, opt.max_QED))

        if opt.conds=="q":
            break
        if opt.conds == "m":
            molecule = inference_phys(opt, model, SRC, TRG,spec_conds=[1,1,0])
            molecule.to_csv('infernece.csv')
            break
        if opt.conds == "s":
            molecule = inference(opt, model, SRC, TRG)
            break
        else:
            molecule = inference(opt, model, SRC, TRG)


if __name__ == '__main__':
    main()
