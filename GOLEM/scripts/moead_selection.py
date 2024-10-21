import numpy as np
from scipy.spatial.distance import cdist


class MOEAD:
    """MOEA/D selection strategy.

    For details see: https://ieeexplore.ieee.org/document/4358754?arnumber=4358754
    """

    def __init__(self, single_demention_selection, init_pop, moead_n_neighbors, *args, **kwargs):
        self.ref_dirs, self.ideal, self.neighbors = self._setup(init_pop, moead_n_neighbors)
        self.single_demention_selection = single_demention_selection

    def __call__(self, pop, pop_size, **kwargs):
        """Selects best individuals."""
        pop = self._set_moead_fitness(pop)
        selected = self.single_demention_selection(pop, pop_size)
        return selected

    def _setup(self, pop, n_neighbors=2):
        ref_dirs = self._get_uniform_weight(pop, len(pop[0].fitness.values))
        ideal = np.min([ind.fitness for ind in pop], axis=0)
        neighbors = np.argsort(cdist(ref_dirs, ref_dirs), axis=1, kind='quicksort')[
            :,
            :n_neighbors,
        ]
        return ref_dirs, ideal, neighbors

    def _set_moead_fitness(self, pop):
        for j, ind in enumerate(pop):
            max_fun = -1.0e30
            for n in range(len(ind.fitness)):
                diff = abs(ind.fitness[n] - self.ideal[n])
                if self.ref_dirs[j][n] == 0:
                    feval = 0.0001 * diff
                else:
                    feval = diff * self.ref_dirs[j][n]

                if feval > max_fun:
                    max_fun = feval

            ind.fitness = [max_fun + len(self.neighbors[j])]

        return pop

    def _get_uniform_weight(self, pop, n_obj):
        assert n_obj == 2 or n_obj == 3, 'MOEAD can handle only 2 or 3 objectives problems'
        m = len(pop)
        if n_obj == 2:
            ref_dirs = [[None for _ in range(n_obj)] for i in range(m)]
            for n in range(m):
                a = 1.0 * float(n) / (m - 1)
                ref_dirs[n][0] = a
                ref_dirs[n][1] = 1 - a
        elif n_obj == 3:
            m = len(pop)

            ref_dirs = []
            for i in range(m):
                for j in range(m):
                    if i + j <= m:
                        k = m - i - j
                        weight_scalars = [None] * 3
                        weight_scalars[0] = float(i) / (m)
                        weight_scalars[1] = float(j) / (m)
                        weight_scalars[2] = float(k) / (m)
                        ref_dirs.append(weight_scalars)
            # Trim number of weights to fit population size
            ref_dirs = sorted((x for x in ref_dirs), key=lambda x: sum(x), reverse=True)
            ref_dirs = ref_dirs[:m]

        return ref_dirs
