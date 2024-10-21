import argparse
import os
import pickle
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from guacamol.score_modifier import ScoreModifier
from guacamol.scoring_function import InvalidMolecule, ScoringFunction

from GraphGA_baseline.graphga import GB_GA_Generator

import_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(import_path)

import numpy as np
import pandas as pd
from rdkit import RDConfig
from rdkit.Chem.rdchem import BondType

from GraphGA_baseline.scripts.mol_metrics import CocrystalsMetrics
from GraphGA_baseline.scripts.utils import project_root

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))


class CCScoringFunction(ScoringFunction):

    def __init__(self, score_modifier: ScoreModifier = None, drug=None) -> None:
        """
        Args:
            score_modifier: Modifier to apply to the score. If None, will be LinearModifier()
        """
        self.drug = drug
        super().__init__(score_modifier=score_modifier)

    def score(self, smiles: str) -> [float, float, float]:
        try:
            met = [-CocrystalsMetrics(drug).orthogonal_planes(smiles),
                   -CocrystalsMetrics(drug).unobstructed(smiles),
                   -CocrystalsMetrics(drug).h_bond_bridging(smiles)]
            return met
        except InvalidMolecule:
            return [self.corrupt_score, self.corrupt_score, self.corrupt_score]
        except Exception:
            print(f'Unknown exception thrown during scoring of {smiles}')
            return [self.corrupt_score, self.corrupt_score, self.corrupt_score]

    def score_list(self, smiles_list: List[str]) -> List[float]:
        return [self.score(smiles) for smiles in smiles_list]

    @abstractmethod
    def raw_score(self, smiles: str) -> float:
        """
        Get the objective score before application of the modifier.

        For invalid molecules, `InvalidMolecule` should be raised.
        For unsuccessful score calculations, `ScoreCannotBeCalculated` should be raised.
        """
        raise NotImplementedError


def run_experiment(optimizer_setup: Callable = None,
                   optimizer_cls=None,
                   adaptive_kind=None,
                   max_heavy_atoms: int = 50,
                   atom_types: Optional[List[str]] = None,
                   bond_types: Sequence[BondType] = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE),
                   initial_data_path: Optional[str] = None,
                   col_name: str = 'generated_coformers',
                   pop_size: int = 20,
                   num_trials: int = 1,
                   trial_timeout: Optional[int] = None,
                   trial_iterations: Optional[int] = None,
                   visualize: bool = False,
                   save_history: bool = True,
                   drug='CN1C2=C(C(=O)N(C1=O)C)NC=N2',
                   result_dir: Optional[str] = None,
                   pretrain_dir: Optional[str] = None
                   ):
    optimizer_id = 'GraphGA'

    experiment_id = f'Experiment [optimizer={optimizer_id} pop_size={pop_size}]'
    exp_name = f'init1'
    result_dir = Path(result_dir) / exp_name

    initial_smiles = pd.read_csv(initial_data_path)[col_name]
    initial_molecules = []
    for smiles in initial_smiles:
        try:
            mol = smiles  # MolGraph.from_smiles(smiles)
            initial_molecules.append(mol)
        except Exception as ex:
            print(ex)
            continue

    trial_results = []
    trial_histories = []

    scoring_function = CCScoringFunction(drug=drug)
    from joblib.externals.loky.backend.context import set_start_method

    set_start_method('spawn')
    for trial in range(6, num_trials):
        np.random.seed(trial)
        optimiser = GB_GA_Generator(smi_file='./guacamol_v1_all.smiles',
                                    population_size=pop_size,
                                    offspring_size=pop_size,
                                    generations=trial_iterations,
                                    mutation_rate=0.01,
                                    n_jobs=-1)

        found_graphs = optimiser.generate_optimized_molecules(scoring_function=scoring_function,
                                                              number_molecules=1,
                                                              starting_population=None)

        print(found_graphs)
        history = optimiser.history

        trial_histories.append(history)

        with open(f"smiles_history_{trial}.pkl", "wb") as fp:  # Pickling
            pickle.dump(optimiser.smiles_history, fp)

        with open(f"full_history_{trial}.pkl", "wb") as fp:  # Pickling
            pickle.dump(optimiser.full_history, fp)

        with open(f"fitn_history_{trial}.pkl", "wb") as fp:  # Pickling
            pickle.dump(optimiser.history, fp)

        with open(f"smiles_final_{trial}.pkl", "wb") as fp:  # Pickling
            pickle.dump(found_graphs, fp)

    # Compute mean & std for metrics of trials
    # ff = max(optimiser.population_scores)
    trial_metrics = np.array([fit for fit in trial_results])
    trial_metrics_mean = trial_metrics.mean(axis=0)
    trial_metrics_std = trial_metrics.std(axis=0)
    print(f'Experiment {experiment_id}\n'
          f'finished with metrics:\n'
          f'mean={trial_metrics_mean}\n'
          f' std={trial_metrics_std}')
    # return trial_histories
    return exp_name


def visualize_run_results(molecules,
                          objective,
                          history,
                          save_path: Path,
                          show: bool = False):
    save_path.mkdir(parents=True, exist_ok=True)


def parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-drug', type=str, default='CN1C2=C(C(=O)N(C1=O)C)NC=N2')
    parser.add_argument('-data_paths', nargs='+',
                        default=os.path.join(project_root(), 'pipeline/result/GAN_all_valid.csv'), help='')
    parser.add_argument('-coformers_col_name', type=str, default='generated_coformers')
    parser.add_argument('-results_folder', type=str, default='GEMOL/results')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parameter()
    drug = args.drug
    data_paths = args.data_paths
    col_name = args.coformers_col_name
    results_folder = args.results_folder
    if not isinstance(data_paths, list):
        data_paths = [data_paths]

    exp_ids = []
    for init_path in data_paths:
        exp_id = run_experiment(initial_data_path=init_path,
                                col_name=col_name,
                                max_heavy_atoms=50,
                                trial_timeout=60,
                                trial_iterations=200,
                                pop_size=200,
                                visualize=True,
                                num_trials=10,
                                drug=drug,
                                result_dir=results_folder,
                                adaptive_kind=None,
                                pretrain_dir=None
                                )
        exp_ids.append(exp_id)
        exp_results_folder = os.path.join(results_folder, exp_id)
