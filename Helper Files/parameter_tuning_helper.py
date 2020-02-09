# -*- coding: utf-8 -*-

###########################################################################
"""
These are helper functions are specifically created for the loan dataset.

https://koremarcel.com
"""
############################################################################
from numpy.random import RandomState
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# classification algorithms
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# hyper-parameter tuning
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.fmin import fmin
import catboost as cb

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


def lgb_tuning(x_train, y_train, x_test, y_test,
               cpu_count,
               scoring,
               cv_value,
               max_evals_num,
               device_type,
               categorical_features,
               early_stopping_rounds=100,
               class_weights='balanced'
               ):
    """

    This function tuning a LightGBM Classification model using the HyperOpt bayesian library.

    :param x_train: training set  data
    :param y_train: target variable
    :param x_test: test test data
    :param y_test: test set target variable
    :param cpu_count: number of cpu cores to use
    :param scoring: scoring metric to be used for the tuning
    :param cv_value: number of cross validations to perform
    :param max_evals_num: max evaluations for the hyperopt function
    :param device_type: training on CPU vs GPU
    :param categorical_features: list of categorical features
    :param early_stopping_rounds: # of rounds to stop training if no increase in model performance
    :param class_weights: for imbalanced data sets -
    :return: A tuned LightGBM Classifier.
    """

    start_time = datetime.now()

    if not isinstance(x_train, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")
    if not isinstance(y_train, pd.Series):
        raise ValueError("Object passed is not a dataframe")

    def org_results(trials, hyperparams, model_name):
        fit_idx = -1
        for idx, fit in enumerate(trials):
            hyp = fit['misc']['vals']
            xgb_hyp = {key: [val] for key, val in hyperparams.items()}
            if hyp == xgb_hyp:
                fit_idx = idx
                break

        train_time = str(trials[-1]['refresh_time'] - trials[0]['book_time'])
        score_param = round(trials[fit_idx]['result']['scoring_metric'], 3)
        train_score = round(trials[fit_idx]['result']['train score'], 3)
        test_score = round(trials[fit_idx]['result']['test score'], 3)

        results = {
            'model': model_name,
            'parameter search time': train_time,
            'scoring_metric': score_param,
            'test score': test_score,
            'train score': train_score,
            'parameters': hyperparams
        }
        return results

    def objective(space):

        lgbm = lgb.LGBMClassifier(
            learning_rate=space['learning_rate'],
            n_estimators=int(space['n_estimators']),
            max_depth=int(space['max_depth']),
            num_leaves=int(space['num_leaves']),
            colsample_bytree=space['colsample_bytree'],
            feature_fraction=space['feature_fraction'],
            reg_lambda=space['reg_lambda'],
            categorical_list=space['categorical_list'],
            reg_alpha=space['reg_alpha'],
            min_split_gain=space['min_split_gain'],
            n_jobs=cpu_count,
            device_type=device_type,
            objective='binary',
            class_weight=class_weights
        )

        lgbm.fit(x_train, y_train,
                 eval_set=[(x_train, y_train), (x_test, y_test)],
                 eval_metric=scoring,
                 early_stopping_rounds=early_stopping_rounds,
                 verbose=False
                 )

        # see if you can add stratifiedkfold here

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=lgbm,
                                     X=x_train,
                                     y=y_train,
                                     cv=cv_value,
                                     n_jobs=cpu_count,
                                     scoring=scoring)

        predictions = lgbm.predict(x_test)
        test_preds = lgbm.predict_proba(x_test)[:, 1]
        train_preds = lgbm.predict_proba(x_train)[:, 1]

        train_score = average_precision_score(y_train, train_preds)
        test_score = average_precision_score(y_test, test_preds)
        scoring_metric = average_precision_score(y_test, predictions)

        return {'status': STATUS_OK,
                'loss': 1 - test_score,
                'scoring_metric': scoring_metric,
                'test score': test_score,
                'train score': train_score}

    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
        'n_estimators': hp.quniform('n_estimators', 50, 1200, 25),
        'max_depth': hp.quniform('max_depth', 1, 15, 1),
        'num_leaves': hp.quniform('num_leaves', 10, 150, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
        'feature_fraction': hp.uniform('feature_fraction', .3, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'min_split_gain': hp.uniform('min_split_gain', 0.0001, 0.1),
        'categorical_list': hp.choice('categorical_list', [None, categorical_features])
    }

    trials = Trials()
    lgb_hyperparams = fmin(fn=objective,
                           space=space,
                           algo=tpe.suggest,
                           max_evals=max_evals_num,
                           trials=trials)

    # Fitting LightGBM to the Training set
    light_gbm = lgb.LGBMClassifier(
        learning_rate=round((lgb_hyperparams['learning_rate']), 3),
        n_estimators=int(lgb_hyperparams['n_estimators']),
        max_depth=int(lgb_hyperparams['max_depth']),
        num_leaves=int(lgb_hyperparams['num_leaves']),
        colsample_bytree=lgb_hyperparams['colsample_bytree'],
        feature_fraction=lgb_hyperparams['feature_fraction'],
        reg_lambda=lgb_hyperparams['reg_lambda'],
        reg_alpha=lgb_hyperparams['reg_alpha'],
        min_split_gain=lgb_hyperparams['min_split_gain'],
        n_jobs=cpu_count,
        categorical_list=lgb_hyperparams['categorical_list'],
        objective='binary',
        device_type=device_type,
        class_weight=class_weights
    )

    lgb_results = org_results(trials.trials, lgb_hyperparams, 'LightGBM')

    time_elapsed = datetime.now() - start_time

    print('\n')
    # print(lgb_results)
    # print('\n')
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    return light_gbm


def catboost_tuning(x_train, y_train,
                    x_test, y_test,
                    early_stopping_rounds,
                    cpu_count,
                    scoring,
                    cv_value,
                    max_evals_num,
                    categorical_features,
                    catboost_eval_metric,
                    task_type):

    """
    This function performs parameter tuning a CatBoost Classifier.

    :param x_train: training set  data
    :param y_train: target variable
    :param x_test: test test data
    :param y_test: test set target variable
    :param cpu_count: number of cpu cores to use
    :param scoring: scoring metric to be used for the tuning
    :param cv_value: number of cross validations to perform
    :param max_evals_num: max evaluations for the hyperopt function
    :param catboost_eval_metric: catboost has specific eval metrics i.e., "F1" equivalent to f1_weighted score
    :param categorical_features: list of categorical features
    :param task_type: GPU vs CPU
    :param early_stopping_rounds: # of rounds to stop training if no increase in model performance
    :return: A tuned Catboost Classifier.
    """

    start_time = datetime.now()

    def objective(space):
        cat_classifier = cb.CatBoostClassifier(n_estimators=space['n_estimators'],
                                               depth=int(space['depth']),
                                               l2_leaf_reg=space['l2_leaf_reg'],
                                               learning_rate=space['learning_rate'],
                                               loss_function='Logloss',
                                               eval_metric=catboost_eval_metric,
                                               cat_features=categorical_features,
                                               task_type=task_type, od_type='Iter', od_wait=100,
                                               verbose=False
                                               )

        cat_classifier.fit(x_train, y_train,
                           eval_set=(x_test, y_test),
                           early_stopping_rounds=early_stopping_rounds)

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=cat_classifier,
                                     X=x_train,
                                     y=y_train,
                                     cv=cv_value,
                                     n_jobs=cpu_count,
                                     scoring=scoring)

        cross_val_mean = accuracies.mean()

        return {'loss': 1 - cross_val_mean, 'status': STATUS_OK}

    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
        'depth': hp.quniform('depth', 1, 15, 1),
        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),
        'n_estimators': hp.choice('n_estimators', range(20, 205, 5))
    }

    trials = Trials()

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals_num,
                trials=trials,
                rstate=RandomState(123)
                )

    catboost_classifier = cb.CatBoostClassifier(n_estimators=best['n_estimators'],
                                                depth=int(best['depth']),
                                                learning_rate=best['learning_rate'],
                                                loss_function='Logloss',
                                                l2_leaf_reg=best['l2_leaf_reg'],
                                                eval_metric=catboost_eval_metric,
                                                cat_features=categorical_features,
                                                task_type=task_type,
                                                od_type='Iter',
                                                od_wait=100,
                                                verbose=False)

    time_elapsed = datetime.now() - start_time

    print(best)
    print('\n')

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    return catboost_classifier


def xgboost_parameter_tuning(x_train, y_train,
                             x_test, y_test,
                             cpu_count,
                             scoring,
                             early_stopping_rounds,
                             cv_value,
                             max_evals_num,
                             ):
    """
    This function performs parameter tuning a XGBoost Classifier.

    :param x_train: training set  data
    :param y_train: target variable
    :param x_test: test test data
    :param y_test: test set target variable
    :param cpu_count: number of cpu cores to use
    :param scoring: scoring metric to be used for the tuning
    :param cv_value: number of cross validations to perform
    :param max_evals_num: max evaluations for the hyperopt function
    :param early_stopping_rounds: # of rounds to stop training if no increase in model performance
    :return: A tuned XGBoost Classifier.
    """

    start_time = datetime.now()

    def objective(space):
        xgb_clf = xgb.XGBClassifier(n_estimators=space['n_estimators'],
                                    max_depth=int(space['max_depth']),
                                    learning_rate=space['learning_rate'],
                                    gamma=space['gamma'],
                                    min_child_weight=space['min_child_weight'],
                                    subsample=space['subsample'],
                                    colsample_bytree=space['colsample_bytree'],
                                    n_jobs=cpu_count,
                                    )

        eval_set = [(x_test, y_test)]
        xgb_clf.fit(x_train, y_train,
                    eval_set=eval_set,
                    verbose=False,
                    early_stopping_rounds=early_stopping_rounds)

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=xgb_clf,
                                     X=x_train,
                                     y=y_train,
                                     cv=cv_value,
                                     n_jobs=cpu_count,
                                     scoring=scoring)
        cross_val_mean = accuracies.mean()

        return {'loss': 1 - cross_val_mean, 'status': STATUS_OK}

    space = {
        'max_depth': hp.choice('max_depth', range(5, 30, 1)),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.5, 0.01),
        'n_estimators': hp.choice('n_estimators', range(20, 205, 5)),
        'gamma': hp.quniform('gamma', 0, 0.50, 0.01),
        'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
    }

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals_num,
                trials=trials)

    classifier = xgb.XGBClassifier(n_estimators=best['n_estimators'],
                                   max_depth=best['max_depth'],
                                   learning_rate=best['learning_rate'],
                                   gamma=best['gamma'],
                                   min_child_weight=best['min_child_weight'],
                                   subsample=best['subsample'],
                                   colsample_bytree=best['colsample_bytree'],
                                   n_jobs=cpu_count
                                   )

    time_elapsed = datetime.now() - start_time

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    return classifier


def lr_tuning(x_train, y_train,
              cpu_count,
              class_weights,
              scoring,
              max_evals_num,
              cv_value
              ):

    """
    This function tunes a logistic regression classifier

    :param x_train: training set data
    :param y_train: target feature
    :param cpu_count: # of CPU Scores
    :param class_weights: class weight value for imbalanced dataset
    :param scoring: Scoring metric to use
    :param max_evals_num:
    :param cv_value: # of cross validation
    :return: A tuned logistic regression classifier
    """

    if not isinstance(x_train, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")
    if not isinstance(y_train, pd.Series):
        raise ValueError("Object passed is not a dataframe")

    def objective(space):

        # Logistic Regression
        classifier = LogisticRegression(
            class_weight=class_weights,
            n_jobs=cpu_count,
            solver=space['solver']
        )

        classifier.fit(x_train, y_train)

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=classifier,
                                     X=x_train,
                                     y=y_train,
                                     cv=cv_value,
                                     n_jobs=cpu_count,
                                     scoring=scoring)

        cross_val_mean = accuracies.mean()

        return {'loss': 1 - cross_val_mean, 'status': STATUS_OK}

    space = {
        'solver': hp.choice('solver', ['lbfgs','liblinear'])
    }

    trials = Trials()

    lr_hyperparams = fmin(fn=objective,
                          space=space,
                          algo=tpe.suggest,
                          max_evals=max_evals_num,
                          trials=trials)

    # Fitting LightGBM to the Training set

    lr_clf = LogisticRegression(
            class_weight=class_weights,
            n_jobs=cpu_count,
            solver=lr_hyperparams['solver']
        )

    return lr_clf


def knn_tuning(x_train, y_train,
               cpu_count,
               scoring,
               max_evals_num,
               cv_value
               ):
    """
    This function tunes a k-nearest neighbor classifier
    :param x_train: training dataset
    :param y_train: target feature
    :param cpu_count: # of CPU Cores
    :param scoring: Scoring metric to be used
    :param max_evals_num: Maximum number of evaluations for hyperopt
    :param cv_value: # of cross validation folds
    :return: A tuned knn classifier
    """

    if not isinstance(x_train, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")
    if not isinstance(y_train, pd.Series):
        raise ValueError("Object passed is not a dataframe")

    def objective(space):

        classifier = KNeighborsClassifier(
            n_neighbors=space['n_neighbors'],
            n_jobs=cpu_count
        )

        classifier.fit(x_train, y_train)

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=classifier,
                                     X=x_train,
                                     y=y_train,
                                     cv=cv_value,
                                     n_jobs=cpu_count,
                                     scoring=scoring)

        cross_val_mean = accuracies.mean()

        return {'loss': 1 - cross_val_mean, 'status': STATUS_OK}

    space = {
        'n_neighbors': hp.choice('n_neighbors', range(3,100)),
    }

    trials = Trials()

    knn_hyperparams = fmin(fn=objective,
                           space=space,
                           algo=tpe.suggest,
                           max_evals=max_evals_num,
                           trials=trials)

    # Fitting LightGBM to the Training set

    knn_clf = KNeighborsClassifier(
        n_neighbors=int(knn_hyperparams['n_neighbors']),
        n_jobs=cpu_count
    )

    return knn_clf


def rf_tuning(x_train, y_train,
              cpu_count, scoring,
              cv_value, max_evals_num):
    """
    This function tunes a Random Forest Classifier

    :param x_train: training dataset
    :param y_train: target feature
    :param cpu_count: # of CPU Cores
    :param scoring: Scoring metric to be used
    :param cv_value: cross validation
    :param max_evals_num: # of evaluations for hyperopt
    :return: A tuned Random Forest Classifier
    """

    start_time = datetime.now()

    def objective(space):

        classifier = RandomForestClassifier(n_estimators=space['n_estimators'],
                                            max_depth=int(space['max_depth']),
                                            criterion=space['criterion'],
                                            n_jobs=cpu_count,
                                            class_weight='balanced'
                                            )

        classifier.fit(x_train, y_train)

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator=classifier,
                                     X=x_train,
                                     y=y_train,
                                     cv=cv_value,
                                     n_jobs=cpu_count,
                                     scoring=scoring)
        cross_val_mean = accuracies.mean()

        return {'loss': 1 - cross_val_mean, 'status': STATUS_OK}

    space = {
        'max_depth': hp.choice('max_depth', range(1, 20)),
        'n_estimators': hp.choice('n_estimators', range(1, 20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"])
    }

    trials = Trials()
    rf_hyperparams = fmin(fn=objective,
                          space=space,
                          algo=tpe.suggest,
                          max_evals=max_evals_num,
                          trials=trials)

    rf_classifier = RandomForestClassifier(n_estimators=rf_hyperparams['n_estimators'],
                                           max_depth=rf_hyperparams['max_depth'],
                                           criterion=rf_hyperparams['criterion'],
                                           n_jobs=cpu_count,
                                           class_weight='balanced'
                                           )

    time_elapsed = datetime.now() - start_time

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    return rf_classifier