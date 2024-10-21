
#This is an example of code to start generating molecules.

# n_samples - number of molecules to be generated.
# path_to_save - suffix to the file path to save the molecules.
# It is necessary to give a name to the file for generation.
# save - whether to save the generated molecules? True/False
# spec_conds - None for random assignment of physical properties/
# list for specifying molecules of interest. Example: [1,1,0].
# The generator function returns a dataset with valid generated column format molecules: ['unobstructed', 'orthogonal_planes', 'h_bond_bridging', 0, 'sa'].
# where 'unobstructed', 'orthogonal_planes', 'h_bond_bridging' are physical properties.
# 0 - column with SMILES of molecules.
# sa - sa value for the molecule.

from generate import generator

df = generator(n_samples=10,
               path_to_save='',
               cuda=False,
               save=False,
               spec_conds = [1,1,0])
print(df)