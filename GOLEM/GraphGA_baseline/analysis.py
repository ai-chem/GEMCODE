import os
import pickle
from copy import deepcopy
from statistics import stdev

import numpy as np
import scipy.stats as stats

import pandas as pd
import seaborn as sns
from matplotlib import axis, pyplot as plt

from rdkit import Chem


def get_final_dataset(results_folder='.'):
    best_smiles = dict()

    for trial in range(0, 11):
        try:
            with open(f"smiles_history_{trial}.pkl", "rb") as fp:  # Pickling
                smiles_history = pickle.load(fp)

            with open(f"full_history_{trial}.pkl", "rb") as fp:  # Pickling
                full_history = pickle.load(fp)
        except Exception:
            continue
        from rdkit.Contrib.SA_Score import sascorer

        for i,pop in enumerate(full_history):
            for j, ind in enumerate(pop):
                try:
                    sa = sascorer.calculateScore(Chem.MolFromSmiles(smiles_history[i][j]))
                except TypeError:
                    sa = sascorer.calculateScore(smiles_history[i][j])

                #ind[3] = sa
                if -ind[0] <= -0.332 and -ind[1] <= -0.5 and \
                        -ind[2] <= 0.5 and sa <= 3:
                    best_smiles.update({smiles_history[i][j]: ind})

    result = {'drug': ['CN1C2=C(C(=O)N(C1=O)C)NC=N2'] * len(best_smiles), 'generated_coformers': [],
              'orthogonal_planes': [], 'unobstructed': [], 'h_bond_bridging': [], 'sa_score': []}

    for smiles, ind in best_smiles.items():
        result['unobstructed'].append(abs(ind[1]))
        if isinstance(smiles, str):
            result['generated_coformers'].append(smiles)
        else:
            result['generated_coformers'].append(Chem.MolToSmiles(smiles))

        result['orthogonal_planes'].append(abs(ind[0]))
        result['h_bond_bridging'].append(1 + ind[2])
        try:
            sa = sascorer.calculateScore(Chem.MolFromSmiles(smiles))
        except TypeError:
            sa = sascorer.calculateScore(smiles)
        result['sa_score'].append(sa)

    df = pd.DataFrame.from_dict(result)
    df.to_csv(os.path.join(results_folder, 'all_valid.csv'), index=False)
    return df

def plot_objs(k, ttl):
    average_fitness_per_gen = []
    confidence_fitness_per_gen = []
    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor='w')
    xlabel = 'Generation'
    full_histories = []
    for trial in range(0, 11):
        try:
            with open(f"full_history_{trial}.pkl", "rb") as fp:  # Pickling
                full_histories.append(pickle.load(fp))
        except Exception:
            continue

    for i in range(len(full_histories[0])):
        average_fitness_per_gen_all = []
        average_fitness_per_gen_all_prev = []
        for t in range(0, len(full_histories)):
            average_fitness_per_gen_all_prev = deepcopy(average_fitness_per_gen_all)
            pop = full_histories[t][i]    
            pop_objs = ([-p[k] for p in pop])
            #if len(average_fitness_per_gen_all) == 0 or np.max(pop_objs)<average_fitness_per_gen_all[-1]:
            average_fitness_per_gen_all.append(np.min(pop_objs))
            #for ri, r in enumerate(average_fitness_per_gen_all):
            #    if len(average_fitness_per_gen_all_prev) > 0 and average_fitness_per_gen_all[ri]>average_fitness_per_gen_all_prev[ri-1]:
            #        average_fitness_per_gen_all[ri] = average_fitness_per_gen_all_prev[ri-1]
            #else:
            #    average_fitness_per_gen_all.append(average_fitness_per_gen[-1])
                
        average_fitness_per_gen.append(np.mean(average_fitness_per_gen_all))
        confidence = stdev(average_fitness_per_gen_all) / np.sqrt(len(average_fitness_per_gen_all))
        confidence_fitness_per_gen.append(confidence)
    # Compute confidence interval
    xs = np.arange(len(average_fitness_per_gen))
    ys = np.array(average_fitness_per_gen)
    z_score: float = 1.96
    ci = z_score * np.array(confidence_fitness_per_gen)

    ax.plot(xs, average_fitness_per_gen)
    ax.fill_between(xs, (ys - ci), (ys + ci), alpha=.2)
    ax.set_ylabel('Fitness')
    ax.set_xlabel(xlabel)
    ax.set_title(ttl)
    if k == 0:
        ax.set_ylim(-0.82, -0.45)
    if k == 1:
        ax.set_ylim(-0.962, -0.89)
    if k == 2:
        ax.set_ylim(0.09, 0.25)
    ax.set_xlim(0, 175)
    ax.grid(axis='y')
    plt.legend()
    plt.savefig(f'graphga_{k}.png', dpi=100)

def visualize_results(feature_names, initial_molecules, gemol_result, results_folder):
    for feature_name in feature_names:
        my_pal = {"initial": "darkcyan", "GraphGA": 'paleturquoise'}
        df = pd.DataFrame(data={#'initial': initial_molecules[feature_name],
                                'GraphGA': gemol_result[feature_name]
                                })
        sns.violinplot(df, palette=my_pal)
        pd.set_option('display.max_columns', None)
        print(feature_name)
        statistics = df.describe().T
        print(stats)
        statistics.to_csv(os.path.join(results_folder, f'{feature_name}_stats.csv'))
        plt.xticks([0], ["GraphGA"])#["Initial", "Evo"])
        plt.title(feature_name)
        plt.savefig(os.path.join(results_folder, 'visualisations', f"violins_{feature_name}.png"), dpi=250)
        plt.close()



# def plot_experiment_comparison(experiment_ids: List[str],
#                                metric_ids: Optional[List[int]] = None,
#                                results_dir='./results'):
#     mlp_line = MultipleFitnessLines.from_saved_histories(experiment_ids, root_path=Path(results_dir))
#
#     metric_ids = metric_ids or [0]
#     for metric_id in metric_ids:
#         mlp_line.visualize(metric_id=metric_id, with_confidence=True, save_path=os.path.join(results_dir, f'mlp_line_metric_{metric_id}.png'))
#

def get_statistical_significance(feature_names, initial_molecules, gemol_result, results_folder):
    data = {'feature': [], 'init_median': [], 'evo_median': [], 'significant': [], 'statistic': [], 'pvalue': []}
    for feature in feature_names:
        stat_sign = False
        res = stats.mannwhitneyu(x=initial_molecules[feature], y=gemol_result[feature],
                                 alternative='two-sided')
        if res.pvalue < 0.05:
            res = stats.mannwhitneyu(x=initial_molecules[feature], y=gemol_result[feature],
                                     alternative='less')
            stat_sign = res.pvalue < 0.05
        data['feature'].append(feature)
        data['init_median'].append(initial_molecules[feature].median())
        data['evo_median'].append(gemol_result[feature].median())
        data['significant'].append(stat_sign)
        data['statistic'].append(res.statistic)
        data['pvalue'].append(res.pvalue)
    df = pd.DataFrame(data=data)
    df.to_csv(os.path.join(results_folder, 'stat_significance.csv'))


graphga_result = get_final_dataset()
visualize_results(['unobstructed', 'orthogonal_planes', 'h_bond_bridging', 'sa_score'],
                       None,
                       graphga_result,
                       '.')

plot_objs(1, "Unobstructed planes"),
plot_objs(0, "Orthogonal planes"),
plot_objs(2, "H-bonds bridging")
    #     get_statistical_significance(['unobstructed', 'orthogonal_planes', 'h_bond_bridging'],
    #                                  init_df,
    #                                  gemol_result,
    #                                  exp_results_folder)
    # plot_experiment_comparison(exp_ids, metric_ids=[0, 1, 2, 3], results_dir=results_folder)

