import os
import sys

import pandas as pd
from rdkit import RDConfig

from GOLEM.scripts.mol_graph import MolGraph
from pipeline.classifier import Classifier

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def sa_score(mol_graph: MolGraph) -> float:
    """Synthetic Accessibility score is a metric used to evaluate the ease of synthesizing a molecule.
    The SA score takes into account a variety of factors such as the number of synthetic steps,
    the availability of starting materials, and the feasibility of the reaction conditions required for each step.

    It is ranged between 1 and 10, 1 is the best possible score.
    """

    molecule = mol_graph.get_rw_molecule(aromatic=True)
    score = sascorer.calculateScore(molecule)
    return score


class CocrystalsMetrics:
    def __init__(self, drug: str):
        self.classifier = Classifier()
        self.drug = drug

    def unobstructed(self, generated_coformer: MolGraph):
        clf_data, generated_coformers_clear = self.classifier.create_clf_dataframe(self.drug,
                                                                                   [generated_coformer.get_smiles()])

        clf_data_unobstructed = pd.DataFrame(clf_data[self.classifier.features_unobstructed])
        clf_prediction_unobstructed = self.classifier.gbc_unobstructed.predict_proba(clf_data_unobstructed)
        return -clf_prediction_unobstructed[0][1]

    def orthogonal_planes(self, generated_coformer: MolGraph):
        clf_data, generated_coformers_clear = self.classifier.create_clf_dataframe(self.drug,
                                                                                   [generated_coformer.get_smiles()])

        clf_data_orthogonal_planes = pd.DataFrame(clf_data[self.classifier.features_orthogonal_planes])
        clf_prediction_orthogonal_planes = \
            self.classifier.gbc_orthogonal_planes.predict_proba(clf_data_orthogonal_planes)

        return -clf_prediction_orthogonal_planes[0][1]

    def h_bond_bridging(self, generated_coformer: MolGraph):
        clf_data, generated_coformers_clear = self.classifier.create_clf_dataframe(self.drug,
                                                                                   [generated_coformer.get_smiles()])
        clf_data_h_bond_bridging = pd.DataFrame(clf_data[self.classifier.features_h_bond_bridging])
        clf_prediction_h_bond_bridging = self.classifier.gbc_h_bond_bridging.predict_proba(clf_data_h_bond_bridging)
        return clf_prediction_h_bond_bridging[0][1]
