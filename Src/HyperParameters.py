# import library
import numpy as np
from skopt.space import Real, Integer, Categorical
import json


# all the paramaters
def rfc_parms(class_weight_dict):
    # grid search Hyperparameters Random Forest Classfiier
    parameters = {
        'n_estimators': list(int(round(x)) for x in np.linspace(50, 500, 4)),
        'min_samples_split': list(int(round(x)) for x in np.linspace(10, 50, 4)),
        'max_features': ['sqrt', 'log2'],
        'max_depth': list(int(round(x)) for x in np.linspace(5, 10, 4)),
        'class_weight': ['balanced', 'balanced_subsample', class_weight_dict]
    }
    return parameters

def rfc_search_space():
    # define the search space
    search_space = {
        'n_estimators': Integer(50, 500),
        'max_depth': Integer(5, 10),
        'min_samples_split': Integer(10, 50),
        'max_features': Categorical(['sqrt', 'log2']),
        'class_weight': Categorical(['balanced', 'balanced_subsample'])
        # 'class_weight': Categorical(['balanced', 'balanced_subsample', frozenset(class_weight_dict.items())])     
    }
    return search_space


def lrc_parms(class_weight_dict):
    # grid search Hyperparameters Logistic Regression
    parameters = dict(
    C = list(np.round(np.logspace(np.log10(0.01), np.log10(10), num=10), 2)),
    l1_ratio = list(np.round(np.logspace(np.log10(0.01), np.log10(1), num=10), 2)),
    class_weight = [class_weight_dict, 'balanced']
    )
    return parameters


def lrc_search_space():
    # define the search space
    search_space = {
        'C': Real(0.01, 10.0, prior='log-uniform'),
        'l1_ratio': Real(0.01, 1.0, prior='uniform'),
         'class_weight': Categorical(['balanced', None])
    }
    return search_space


def xgbc_parms(scale_pos_weight):
    # define common parameters
    search_space = {
        'n_estimators': list(int(round(x)) for x in np.linspace(50, 500, 3)),
        'max_depth': list(int(round(x)) for x in np.linspace(3, 10, 3)),
        'learning_rate': list(np.round(x, 2) for x in np.linspace(0.01, 2.0, 3)),
        'subsample': list(np.round(x, 2) for x in np.linspace(0.4, 1.0, 3)),
        'colsample_bytree': list(np.round(x, 2) for x in np.linspace(0.5, 1.0, 3)),
        'gamma': list(np.round(x, 2) for x in np.linspace(0.5, 2.0, 4))
    }

    # add scale_pos_weight based on its value
    if scale_pos_weight > 1:
        search_space['scale_pos_weight'] = [1, scale_pos_weight]
    else:
        search_space['scale_pos_weight'] = [scale_pos_weight, 1]

    return search_space


def xgbc_search_space(scale_pos_weight):
    # Define common search space parameters
    search_space = {
        'n_estimators': Integer(50, 500),
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 2.0, prior='log-uniform'),
        'subsample': Real(0.4, 1.0, prior='uniform'),
        'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
        'gamma': Real(0.5, 4.0, prior='uniform')
    }

    # Add scale_pos_weight based on its value
    if scale_pos_weight > 1:
        search_space['scale_pos_weight'] = Real(1, scale_pos_weight)
    else:
        search_space['scale_pos_weight'] = Real(scale_pos_weight, 1)

    return search_space



def knn_parms():
    # grid search Hyperparameters KNN Classifier
    parameters = {
        'n_neighbors': list(int(round(x)) for x in np.linspace(1, 20, 10)),
        'weights': ['uniform', 'distance'],
        'leaf_size': list(int(round(x)) for x in np.linspace(3, 50, 5)),
        'p': [1,2]
    }
    return parameters


def knn_search_space():
    # define the search space
    search_space = {
        'n_neighbors': Integer(1, 20),
        'weights': Categorical(['uniform', 'distance']),
        'leaf_size': Integer(3, 50),
        'p': Categorical([1, 2])
    }
    return search_space


def ada_parms():
    # grid search Hyperparameters AdaBoost Classifier
    parameters = {
        'n_estimators': list(int(round(x)) for x in np.linspace(20, 500, 5)),
        'learning_rate': list(np.round(x, 2) for x in np.linspace(0.01, 3.0, 7))
    }
    return parameters


def ada_search_space():
    # define the search space
    search_space = {
        'n_estimators': Integer(20, 500),
        'learning_rate': Real(0.01, 3, prior='log-uniform')
    }
    return search_space
