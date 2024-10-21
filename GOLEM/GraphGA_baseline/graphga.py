from __future__ import print_function

import heapq
import random
from copy import deepcopy
from time import time
from typing import List, Optional

import joblib
import numpy as np
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize
from joblib import delayed
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from . import crossover as co, mutate as mu


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights

    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return

    Returns: a list of RDKit Mol (probably not unique)

    """
    # scores -> probs
    csc = [pr + 0.0001 for pr in population_scores]
    sum_scores = sum(csc)
    population_probs = [p / sum_scores for p in csc]
    mating_pool = list(np.random.choice(population_mol, p=population_probs, size=offspring_size - 1, replace=True))
    mating_pool.append(population_mol[np.argmax(population_scores)])
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """

    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation

    Returns:

    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = co.crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mu.mutate(new_child, mutation_rate)
    return new_child


def is_domainated(score1, score2):
    if score1[0] >= score2[0] and score1[1] >= score2[1] and score1[2] >= score2[2]:
        return True
    return False


def score_mol(mol, score_fn):
    return score_fn(Chem.MolToSmiles(mol))


def score_mol_mo(mol, score_fn, pop_scores):
    fit = 0
    score1 = score_fn(Chem.MolToSmiles(mol))
    for score2 in pop_scores:
        if score1 != score2:
            if is_domainated(score1, score2):
                fit += 1
    return fit / (len(pop_scores) - 1)


def sanitize(population_mol):
    new_population = []
    smile_set = set()
    for mol in population_mol:
        if mol is not None:
            try:
                smile = Chem.MolToSmiles(mol)
                if smile is not None and smile not in smile_set:
                    smile_set.add(smile)
                    new_population.append(mol)
            except ValueError:
                print('bad smiles')
    return new_population


class GB_GA_Generator(GoalDirectedGenerator):

    def __init__(self, smi_file, population_size, offspring_size, generations, mutation_rate, n_jobs=-1,
                 random_start=True, patience=5):

        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.smi_file = smi_file
        self.all_smiles = self.load_smiles_from_file(self.smi_file)
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.patience = patience
        self.history = []
        self.full_history = []
        self.smiles_history = []

    def load_smiles_from_file(self, smi_file):
        with open(smi_file) as f:
            return self.pool(delayed(canonicalize)(s.strip()) for s in f)

    def top_k(self, smiles, scoring_function, k):
        joblist = (delayed(scoring_function.score)(s) for s in smiles)
        scores = self.pool(joblist)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:

        if number_molecules > self.population_size:
            self.population_size = number_molecules
            print(f'Benchmark requested more molecules than expected: new population is {number_molecules}')

        # fetch initial population?
        if starting_population is None:
            print('selecting initial population...')
            if self.random_start:
                starting_population = np.random.choice(self.all_smiles, self.population_size)
            else:
                starting_population = self.top_k(self.all_smiles, scoring_function, self.population_size)

        # select initial population
        population_smiles = heapq.nlargest(self.population_size, starting_population, key=scoring_function.score)
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_scores_prep = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)

        self.full_history.append(deepcopy(population_scores_prep))
        self.smiles_history.append(deepcopy(population_smiles))

        population_scores = self.pool(
            delayed(score_mol_mo)(m, scoring_function.score, population_scores_prep) for m in population_mol)

        self.population_scores = deepcopy(population_scores)

        self.history.append(deepcopy(self.population_scores))
        # evolution: go go go!!
        t0 = time()

        patience = 10

        for generation in range(self.generations):

            # new_population
            mating_pool = make_mating_pool(population_mol, population_scores, self.offspring_size)
            offspring_mol = self.pool(
                delayed(reproduce)(mating_pool, self.mutation_rate) for _ in range(self.population_size))

            # add new_population
            population_mol += offspring_mol
            population_mol = sanitize(population_mol)

            # stats
            gen_time = time() - t0
            mol_sec = self.population_size / gen_time
            t0 = time()

            old_scores = population_scores
            population_scores_prep = self.pool(delayed(score_mol)(m, scoring_function.score) for m in population_mol)

            population_scores = self.pool(
                delayed(score_mol_mo)(m, scoring_function.score, population_scores_prep) for m in population_mol)

            population_tuples = list(zip(population_scores, population_mol, population_scores_prep))
            population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:self.population_size]
            population_mol = [t[1] for t in population_tuples]
            population_scores = [t[0] for t in population_tuples]
            population_scores_prep = [t[2] for t in population_tuples]
            self.full_history.append(deepcopy(population_scores_prep))
            self.smiles_history.append(deepcopy(population_mol))

            population_scores = self.pool(
                delayed(score_mol_mo)(m, scoring_function.score, population_scores_prep) for m in population_mol)

            self.population_scores = deepcopy(population_scores)

            self.history.append(deepcopy(self.population_scores))

            # early stopping
            if population_scores == old_scores:
                patience += 1
                print(f'Failed to progress: {patience}')
                if patience >= self.patience:
                    print(f'No more patience, bailing...')
                    break
            else:
                patience = 0

            print(f'{generation} | '
                  f'max: {np.max(population_scores):.3f} | '
                  f'avg: {np.mean(population_scores):.3f} | '
                  f'min: {np.min(population_scores):.3f} | '
                  f'std: {np.std(population_scores):.3f} | '
                  f'sum: {np.sum(population_scores):.3f} | '
                  f'{gen_time:.2f} sec/gen | '
                  f'{mol_sec:.2f} mol/sec')

        # finally
        return [Chem.MolToSmiles(m) for m in population_mol][:number_molecules]
