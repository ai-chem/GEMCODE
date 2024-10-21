# import libraries
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# %%
# set the target property
target_property = 'Orthogonal planes'
# %%
# load data
df = pd.read_csv('./classifier/data/CCDC_descriptors.csv', delimiter=';', decimal=',')
df_property = pd.read_csv('./classifier/data/dataset_' + target_property + '.csv')
# %%
# split the dataset into training and test samples
X_train, X_test, y_train, y_test = train_test_split(df_property, df[target_property], test_size=0.2, random_state=42)

xtd = pd.DataFrame(X_train)
xtd.insert(X_train.shape[1], 'target', y_train, True)

train_data = TabularDataset(xtd)
model = TabularPredictor(problem_type='binary', label='target').fit(train_data, presets='best_quality',
                                                                    time_limit=10 * 60 * 60)

y_pred = (np.asarray(model.predict_proba(X_test))[:, 1] >= 0.302).astype(int)
y_pred_prob = np.asarray(model.predict_proba(X_test))[:, 1]

# %%
# check the metrics
print(classification_report(y_test, y_pred))
