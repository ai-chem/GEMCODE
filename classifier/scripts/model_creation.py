import pandas as pd
import numpy as np
import pickle as pi
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from scipy.stats import loguniform
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

import argparse


class MLPipeline:

    def __init__(self, df_cocrystals, df_ChEMBL_molecules):

        self.df = df_cocrystals

        self.df_for_scaling = df_ChEMBL_molecules

        self.X_descriptors = self.df.drop(['Unobstructed', 'Orthogonal planes', 'H-bonds bridging'], axis=1)

        self.feature_num = (len(self.df.columns) - 3) // 2

        self.model = GradientBoostingClassifier(random_state=42)

        self.min_max_scaler = object
        self.min_max_scaler_old = object

        self.test_samples = tuple()

        self.lr_model = LogisticRegression(max_iter=3000, random_state=42)
        self.knn_model = KNeighborsClassifier()
        self.dtc_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.rfc_model = RandomForestClassifier(random_state=42)
        self.gbc_model = GradientBoostingClassifier(random_state=42)
        self.abc_model = AdaBoostClassifier(random_state=42)
        self.svm_model = LinearSVC(max_iter=3000, random_state=42)
        self.mlpc_model = MLPClassifier(max_iter=3000, random_state=42)
        self.cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    ######################
    ## Create DataFrame ##
    ######################

    def create_features(self, df):
        '''create_features creates new features (sum and average) 
        and adds them to the input df
        
        output: dataframe with added features'''

        list_of_params = df.columns.tolist()

        for feat_idx in range(self.feature_num):
            df[list_of_params[feat_idx] + '_sum'] = \
                df.iloc[:, feat_idx] + df.iloc[:, feat_idx + self.feature_num]
            df[list_of_params[feat_idx] + '_mean'] = \
                (df.iloc[:, feat_idx] + df.iloc[:, feat_idx + self.feature_num]) / 2

        return df

    def create_scaler(self):
        '''create_scaler fits the scalers to the data

        output: self.min_max_scaler (on the ChemBL database - 172 features)
        self.min_max_scaler_old (on ChemBL database - 86 features)'''

        df_for_scaling_sum_amd_mean = self.create_features(self.df_for_scaling)

        self.min_max_scaler_old = MinMaxScaler().fit(self.df_for_scaling.iloc[:, :self.feature_num * 2])

        self.min_max_scaler = MinMaxScaler().fit(df_for_scaling_sum_amd_mean)

        pi.dump(self.min_max_scaler_old, open('./checkpoints/min_max_scaler_old.pkl', 'wb'))

        pi.dump(self.min_max_scaler, open('./checkpoints/min_max_scaler.pkl', 'wb'))

        return self.min_max_scaler, self.min_max_scaler_old

    def create_and_scale_features(self):
        '''create_and_scale_features applies the feature creation function and 
        scales the data from the original dataset of co-crystals
        
        output: self.X_descriptors - modified features'''

        X_sum_amd_mean = self.create_features(self.X_descriptors)

        X_scale = self.min_max_scaler.transform(X_sum_amd_mean)

        self.X_descriptors = pd.DataFrame(X_scale, columns=X_sum_amd_mean.columns)

        return self.X_descriptors

    def learn_model(self, target_property):
        '''learn_model trains the self.model
        
        output: trained self.model'''

        y = self.df[target_property].tolist()

        x_train, x_test, y_train, y_test = train_test_split(self.X_descriptors, y, test_size=0.2, random_state=42)

        self.test_samples = tuple([x_test, y_test])
        self.model.fit(x_train, y_train)

        return self.model

    def select_features_importances(self, percent_of_selected_features):
        '''select_features_importances selects features according to their 
        importance and removes duplicate features from the list
        
        output: list_selected - important features without duplicates '''

        feature_importances, feature_names = self.model.feature_importances_, self.model.feature_names_in_

        feature_names_sorted = [val[1] for val in sorted(zip(feature_importances, feature_names),
                                                         key=lambda x: x[0],
                                                         reverse=True)]

        max_features = len(feature_names_sorted)
        blacklist = []
        whitelist = []
        list_selected = []
        i = 0
        while (len(list_selected) < max_features):
            try:
                value = feature_names_sorted[i]
                if value not in blacklist:
                    list_selected.append(value)
                    if value in whitelist:
                        whitelist.remove(value)
                splitted_value = value.split(sep='_')
                end = splitted_value[-1]
                endings = ['mean', 'sum']
                if (end in endings):
                    main = splitted_value[:-1][0]
                    endings.remove(end)
                    blacklist.extend(['{m}_{e}'.format(m=main, e=tmp) for tmp in endings])
                    blacklist.extend(['{}.1'.format(main), main])
                else:
                    if value[-2:] != '.1':
                        antivalue = '{}.1'.format(value)
                        blacklist.extend(['{m}_{e}'.format(m=value, e=tmp) for tmp in endings])
                    else:
                        antivalue = value[:-2]
                        blacklist.extend(['{m}_{e}'.format(m=antivalue, e=tmp) for tmp in endings])
                    if antivalue not in list_selected and antivalue not in whitelist and antivalue not in blacklist:
                        whitelist.append(antivalue)
                i += 1
            except:
                break

            current_percent_of_selected_features = 100 * (len(list_selected) / max_features)

            if current_percent_of_selected_features >= percent_of_selected_features:
                break
        list_selected.extend(whitelist)

        return list_selected

    def corr_matrix(self, target_property, df):
        '''corr_matrix plots a correlation matrix
        
        output: corr_matrix - dataframe'''

        corr_matrix = df.corr()

        plt.figure(figsize=(16, 13))

        if len(corr_matrix) < 50:
            size = '_small'
            heatmap = sns.heatmap(corr_matrix, annot=True, annot_kws={"size": 6}, cmap='mako')
        else:
            size = '_large'
            heatmap = sns.heatmap(corr_matrix, annot=False, cmap='mako')

        heatmap.set_title(target_property, fontdict={'fontsize': 12}, pad=12)

        plt.savefig('images/' + target_property + '/corr_matrix_' + size + '.png', bbox_inches='tight')

        return corr_matrix

    def select_features(self, target_property):
        '''select_features triggers a parameter prioritisation function and removes 
        correlated parameters, leaving the same parameters for the molecular pair
        
        output: X_scale_selected_df_without_corr - final dataset for predictions'''

        self.learn_model(target_property)

        feature_names_sorted = self.select_features_importances(50)

        X_scale_selected_df = self.X_descriptors.loc[:, feature_names_sorted]

        corr_matrix_full = self.corr_matrix(target_property, X_scale_selected_df)

        upper = corr_matrix_full.where(np.triu(np.ones(corr_matrix_full.shape), k=1).astype(bool))

        to_drop = [X_scale_selected_df.columns.get_loc(column) for column in upper.columns if any(upper[column] > 0.6)]

        X_scale_selected_df_without_corr = X_scale_selected_df.drop(X_scale_selected_df.columns[to_drop], axis=1)

        add_features = []

        for feature in list(X_scale_selected_df.columns[to_drop]):

            if feature[:-2] not in list(X_scale_selected_df.columns[to_drop]) and feature[:-2] in list(
                    X_scale_selected_df_without_corr.columns):
                add_features.append(feature)

            elif feature + '.1' not in list(X_scale_selected_df.columns[to_drop]) and feature + '.1' in list(
                    X_scale_selected_df_without_corr.columns):
                add_features.append(feature)

        X_scale_selected_df_without_corr[add_features] = X_scale_selected_df[add_features]

        self.corr_matrix(target_property, X_scale_selected_df_without_corr)

        with open("result_features/features_" + target_property + ".txt", "w") as f:

            for feature in X_scale_selected_df_without_corr.columns.tolist():
                f.write(feature)
                f.write("\n")

        X_scale_selected_df_without_corr.to_csv('data/dataset_' + target_property + '.csv', index=False)

        return X_scale_selected_df_without_corr

    def create_dataset(self):
        '''create_dataset creates a dataframe for subsequent 
        parameter selection
        
        output: scaled dataframe with features created'''

        self.create_scaler()

        df = self.create_and_scale_features()

        return df

    #######################################
    ## Select, Create and Optimize model ##
    #######################################

    def make_prediction(self, target_property, X, model, threshold='N'):
        '''make_prediction trains the model fed into the function 
        
        output: result of accuracy and f1 score'''

        y = self.df[target_property].tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)

        if threshold == 'Y' and model != self.svm_model:
            y_pred = (model.predict_proba(X_test)[:, 1] >= 0.302).astype(int)

        else:
            y_pred = model.predict(X_test)

        acc_score = accuracy_score(y_test, y_pred)

        res_f1_score = f1_score(y_test, y_pred)

        return acc_score, res_f1_score

    def result_test_models(self, target_property, X, threshold='N'):
        '''result_test_models calls the function make_prediction for each model
        
        output: list of results of accuracy and f1 score'''

        models = [self.lr_model, self.knn_model, self.dtc_model, self.rfc_model,
                  self.gbc_model, self.abc_model, self.svm_model, self.mlpc_model]

        results_accuracy, results_f1 = [], []

        for model in models:
            accuracy, f1 = self.make_prediction(target_property, X, model, threshold)

            results_accuracy.append(accuracy)

            results_f1.append(f1)

        return results_accuracy, results_f1

    def result_df_test_models(self, target_property, X, threshold='N', threshold_old='N'):
        '''result_df_test_models calls function result_test_models for prediction 
        before conversions with dataset (X_old) and after (X)
        
        output: df_results'''

        name_models = ['LR', 'K-NN', 'DT', 'RF', 'GB', 'AB', 'SVM', 'MLP']

        results_accuracy, results_f1 = self.result_test_models(target_property, X, threshold)

        X_old = pd.DataFrame(self.min_max_scaler_old.transform(self.X_descriptors.iloc[:, :self.feature_num * 2]),
                             columns=self.X_descriptors.iloc[:, :self.feature_num * 2].columns)

        results_accuracy_old, results_f1_old = self.result_test_models(target_property, X_old, threshold_old)

        df_results_accuracy = pd.DataFrame(list(zip(name_models, results_accuracy_old, results_accuracy)),
                                           columns=['name', 'x_old', 'x'])

        df_results_accuracy['score'] = ['accuracy'] * 8

        df_results_f1 = pd.DataFrame(list(zip(name_models, results_f1_old, results_f1)), columns=['name', 'x_old', 'x'])
        df_results_f1['score'] = ['f1'] * 8

        df_results = pd.concat([df_results_accuracy, df_results_f1], ignore_index=True)

        return df_results

    def test_models_plots(self, target_property, X, threshold='N', threshold_old='N'):
        """test_models_plots plots df_results

        output: picture"""

        df_results = self.result_df_test_models(target_property, X, threshold, threshold_old)

        f, ax = plt.subplots(figsize=(4, 3), dpi=200)

        sns.set_palette('mako', n_colors=2)

        sns.barplot(x='x', y='name', hue="score", data=df_results, alpha=.3)

        sns.barplot(x='x_old', y='name', hue="score", data=df_results, alpha=.7)

        ax.set(xlim=(0.0, 0.8), ylabel="Model",
               xlabel="Score")

        plt.legend([], [], frameon=False)

        sns.despine(right=True)

        f.suptitle(target_property, fontsize=14)

        plt.savefig('images/' + target_property + '/test_models_' + target_property + '.png', bbox_inches='tight')

    def optimization_random_search(self, target_property, X, threshold='N'):
        """optimization_random_search

        output: hyperparameters"""

        y = self.df[target_property].tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        space = {'learning_rate': loguniform(1e-5, 1),
                 'n_estimators': sp_randInt(10, 250),
                 'subsample': sp_randFloat(0.1, 0.9),
                 'max_depth': sp_randInt(1, 10)}

        search = RandomizedSearchCV(self.model, space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=self.cv,
                                    random_state=42)

        result = search.fit(X_train, y_train)

        print('Best Score of Random Search: %s' % result.best_score_)
        print('Best Hyperparameters of Random Search: %s' % result.best_params_)

    def optimization_grid_search(self, target_property, X, list_learning_rate, list_n_estimators, list_subsample,
                                 list_max_depth):
        """optimization_grid_search

        output: hyperparameters"""

        y = self.df[target_property].tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        space = {'learning_rate': list_learning_rate,
                 'n_estimators': list_n_estimators,
                 'subsample': list_subsample,
                 'max_depth': list_max_depth}

        search = GridSearchCV(self.model, space, scoring='accuracy', n_jobs=-1, cv=self.cv)

        result = search.fit(X_train, y_train)

        print('Best Score of Grid Search: %s' % result.best_score_)
        print('Best Hyperparameters of Grid Search: %s' % result.best_params_)

        results_df = pd.DataFrame(search.cv_results_)
        results_df.to_csv('./checkpoints/grid_search_results/' + target_property + '.csv', index=False)

        return result.best_params_

    def see_model_scores_and_save(self, optimal_hyperparameters, target_property, X, threshold='N'):
        '''see_model_scores_and_save
        
        output: scores and saved model'''

        if optimal_hyperparameters == None:
            optimal_hyperparameters = {'learning_rate': 0.1, 'n_estimators': 100, 'subsample': 1.0, 'max_depth': 3}

        self.model.__init__(learning_rate=optimal_hyperparameters['learning_rate'],
                            max_depth=optimal_hyperparameters['max_depth'],
                            n_estimators=optimal_hyperparameters['n_estimators'],
                            subsample=optimal_hyperparameters['subsample'])

        acc_score, res_f1_score = self.make_prediction(target_property, X, self.model, threshold)

        print('Final accuracy score for ' + target_property + ': ', acc_score)
        print('Final F1 score ' + target_property + ': ', res_f1_score)

        pi.dump(self.model, open('./checkpoints/gbc_' + target_property + '.pkl', 'wb'))


def main(target_property, CCDC_path, ChEMBL_path):
    df1 = pd.read_csv(CCDC_path, delimiter=';', decimal=',')
    df2 = pd.read_csv(ChEMBL_path)

    pipeline = MLPipeline(df_cocrystals=df1, df_ChEMBL_molecules=df2)

    df_descriptors = pipeline.create_dataset()
    df_property = pipeline.select_features(target_property)

    pipeline.test_models_plots(target_property, df_property)

    if target_property == 'Unobstructed':
        optimal_hyperparameters = {'learning_rate': 0.06, 'max_depth': 3, 'n_estimators': 250, 'subsample': 0.9}
        threshold = 'N'
    elif target_property == 'Orthogonal planes':
        optimal_hyperparameters = {'learning_rate': 0.01, 'max_depth': 6, 'n_estimators': 225, 'subsample': 0.5}
        threshold = 'Y'
    elif target_property == 'H-bonds bridging':
        optimal_hyperparameters = {'learning_rate': 0.07, 'max_depth': 2, 'n_estimators': 250, 'subsample': 0.9}
        threshold = 'N'

    pipeline.see_model_scores_and_save(optimal_hyperparameters, target_property, df_property, threshold)


def parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-target_property', type=str, help='',
                        choices=['Unobstructed', 'Orthogonal planes', 'H-bonds bridging'])
    parser.add_argument('-CCDC_path', type=str, default='data/CCDC_descriptors.csv', help='')
    parser.add_argument('-ChEMBL_path', type=str, default='data/ChEMBL_descriptors.csv', help='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parameter()
    main(args.target_property, args.CCDC_path, args.ChEMBL_path)
