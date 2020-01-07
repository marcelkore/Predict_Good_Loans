# -*- coding: utf-8 -*-

###########################################################################
"""
These are helper functions are specifically created for the loan dataset.

They can be obviously modified for another type of dataset.

https://koremarcel.com
"""
############################################################################

import pandas as pd
# plot classification report
from yellowbrick.classifier import ClassificationReport
from datetime import datetime
from matplotlib.ticker import FuncFormatter
import scikitplot as skplt
import xgboost as xgb
import catboost as cb
from yellowbrick.classifier import ConfusionMatrix

from catboost import Pool, CatBoostClassifier

from yellowbrick.classifier import ROCAUC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from yellowbrick.classifier import PrecisionRecallCurve

# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold

# Model Interpretation
import eli5
from eli5.sklearn import PermutationImportance

# metrics
import scikitplot as skplt
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# sklearn librarie

from sklearn.ensemble import RandomForestClassifier

import scikitplot as skplt

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# classification algorithms
import lightgbm as lgb

# hyper-parameter tuning
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.fmin import fmin
import warnings
warnings.filterwarnings('ignore')

import matplotlib.style as style
import matplotlib.pyplot as plt
style.use('fivethirtyeight')


def calculate_feature_importance(model, x_test, y_test):
    """
    :param model: pass the model to fitted model that you want to plot a learning curve for
    :param x_test: pass a dataframe the training data
    :param y_test: pass the training data target feature
    :return: a data frame containing features and their ranking of importance
    """
    start_time = datetime.now()

    model_name = type(model).__name__

    perm = PermutationImportance(model, random_state=2019).fit(x_test, y_test)
    time_elapsed = datetime.now() - start_time

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    return eli5.show_weights(perm, feature_names=x_test.columns.tolist())


def train_model(model, x_train, y_train, n_jobs, scoring_metric, number_of_folds):
    """
    :param model: pass the classification or regression model
    :param x_train: pass the training data
    :param y_train: pass the target feature
    :param n_jobs: # of cpus to use
    :param number_of_folds: pass the number of folds for cross validation: default = 6
    :param scoring_metric: pass the scoring metric
    :return: returns the training score model score
    """

    start_time = datetime.now()
    stratified_k_fold = StratifiedKFold(n_splits=number_of_folds)
    scoring = scoring_metric
    model_scores = cross_val_score(model, x_train, y_train, cv=stratified_k_fold, scoring=scoring, n_jobs=n_jobs)

    model_name = type(model).__name__

    rounded_model_score = round(model_scores.mean() * 100.0, 3)

    print("Cross Validation {} Training Score for {} is {}".format(scoring_metric, model_name, rounded_model_score))

    time_elapsed = datetime.now() - start_time

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


def plot_learning_curve(model, x_train, y_train, scoring, cpu_count):
    """
    :param model: pass the model to fitted model that you want to plot a learning curve for
    :param x_train: pass the training data
    :param y_train: pass the training data target feature
    :param cpu_count: # of threads to use for the processing
    :param scoring: pass the scoring metric. Default to 'recall'
    :return: a plot of the training data learning curve
    """
    start_time = datetime.now()

    model_name = type(model).__name__

    skplt.estimators.plot_learning_curve(model, x_train, y_train, figsize=(8, 8), n_jobs=cpu_count, scoring=scoring)
    plt.title(scoring);
    plt.show();

    time_elapsed = datetime.now() - start_time

    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    
    
def plot_confusion_matrix (X_train, y_train, X_test, y_test, model, encoder):
    """
   
   
   
   
    """
    encoder = encoder
    
    # The ConfusionMatrix visualizer taxes a model
    cm = ConfusionMatrix(model, encoder=encoder)

    # Fit fits the passed model. This is unnecessary if you pass the visualizer a pre-fitted model
    cm.fit(X_train, y_train)

    # To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
    # and then creates the confusion_matrix from scikit-learn.
    cm.score(X_test, y_test)

    cm.show();

    

def plot_precision_recall_curve_1(X_train, y_train, X_test, y_test, model):
    """
    Plot precision-recall curve to evaluate classification.

    Parameters:
        - y_test: The true values for y
        - preds: The predicted values for y as probabilities
        - positive_class: The label for the positive class in the data
        - ax: The matplotlib Axes object to plot on

    Returns: Plotted precision-recall curve.
    """
     
    
    viz = PrecisionRecallCurve(model)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()


def visual_model_selection(X, y, estimator):
    
    visualizer = ClassificationReport(estimator, classes=['Low', 'Medium', 'High'], cmap='PRGn')
    visualizer.fit(X, y)  
    visualizer.score(X, y)
    visualizer.poof()


def plot_roc_auc_curve(y_pred, y_test, model=None,):
    y_pred = np.concatenate(((1 - y_pred).reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)

    model_title = model_name = type(model).__name__ + ' ROC_CURVE'
    # calculate ROC curves
    skplt.metrics.plot_roc(y_test, y_pred, figsize=(8, 8),title=model_title)
    plt.show();


def plot_precision_recall_curve_2(y_pred, y_test, model=None, ):
    y_pred = np.concatenate(((1 - y_pred).reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1)

    model_title = model_name = type(model).__name__ + ' ROC_CURVE'
    # calculate ROC curves
    skplt.metrics.plot_precision_recall(y_test, y_pred, figsize=(8, 8), title=model_title)
    plt.show();


def plotLiftChart(actual, predicted):
    # https://github.com/reiinakano/scikit-plot/issues/74
    df_dict = {'actual': list (actual), 'pred': list(predicted)}
    df = pd.DataFrame(df_dict)
    pred_ranks = pd.qcut(df['pred'].rank(method='first'), 100, labels=False)
    actual_ranks = pd.qcut(df['actual'].rank(method='first'), 100, labels=False)
    pred_percentiles = df.groupby(pred_ranks).mean()
    actual_percentiles = df.groupby(actual_ranks).mean()
    plt.title('Lift Chart')
    plt.plot(np.arange(.01, 1.01, .01), np.array(pred_percentiles['pred']),
             color='darkorange', lw=2, label='Prediction')
    plt.plot(np.arange(.01, 1.01, .01), np.array(pred_percentiles['actual']),
             color='navy', lw=2, linestyle='--', label='Actual')
    plt.ylabel('Target Percentile')
    plt.xlabel('Population Percentile')
    plt.xlim([0.0, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="best")


def calculate_scale_pos_weight(y_test):
    target_feature_dataframe = y_test.value_counts().to_frame()

    class_1 = target_feature_dataframe.iloc[1].to_list()
    class_2 = target_feature_dataframe.iloc[0].to_list()

    if class_1 > class_2:
        numerator = class_1[0]
        return round((numerator / class_2[0]), 2)
    else:
        numerator = class_2[0]
        return round((numerator / class_1[0]), 2)