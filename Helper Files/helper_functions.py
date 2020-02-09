# -*- coding: utf-8 -*-

###########################################################################
"""
These are helper functions are specifically created for the loan dataset.

They can be obviously modified for another type of dataset.

https://koremarcel.com
"""
############################################################################

import pandas as pd
from datetime import datetime
from yellowbrick.model_selection import LearningCurve
from sklearn.model_selection import StratifiedKFold
import itertools
from sklearn.feature_selection import RFECV
import catboost as cb

# Viz Libraries
import seaborn as sns
sns.set(style='darkgrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
import warnings
import matplotlib.gridspec as gs

from yellowbrick.model_selection import ValidationCurve
import matplotlib as mpl
from cycler import cycler


import matplotlib.style as style
import matplotlib.pyplot as plt
import boruta_py as bpy
from yellowbrick.classifier import DiscriminationThreshold


style.use('fivethirtyeight')
# %matplotlib inline

# metrics
import numpy as np
from sklearn.metrics import roc_curve

# sklearn libraries
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# category encoding
import category_encoders as ce

# classifier
from sklearn.ensemble import RandomForestClassifier

# feature selection
from sklearn.model_selection import learning_curve
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Function to time how long  jobs run
# credit - https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def define_optimal_threshold(model,X_test,y_test,plot=True):
    """
    
    credit: https://www.kaggle.com/deepanshu08/prediction-of-lendingclub-loan-defaulters
    """
    
    fp, tp, threshold = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    
    if plot==True:
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,6));

        ax[0].plot(threshold, tp + (1 - fp));
        ax[0].set_xlabel('Threshold');
        ax[0].set_ylabel('Sensitivity + Specificity');

        ax[1].plot(threshold, tp, label="tp");
        ax[1].plot(threshold, 1 - fp, label="1 - fp");
        ax[1].legend();
        ax[1].set_xlabel('Threshold');
        ax[1].set_ylabel('True Positive & False Positive Rates');

    # finding the optimal threshold for the model
    function = tp + (1 - fp)
    index = np.argmax(function)

    optimal_threshold = round((threshold[np.argmax(function)]),2)
    print("optimal threshold:{}".format(optimal_threshold))
    
    return optimal_threshold


def plot_optimal_threshold(model, x_train, y_train):
    """

    """

    # Visualization Threshold
    visualizer = DiscriminationThreshold(model)

    visualizer.fit(x_train, y_train)  # Fit the data to the visualizer
    visualizer.show();
        

def plot_predicted_probabilities(probabilities):
    """
    
    """
    plt.rcParams['font.size'] = 14
    plt.rcParams["figure.figsize"] = [7,7]
    plt.hist(probabilities)
    plt.xlim(0, 1)
    plt.title('Histogram of predicted probabilities')
    plt.xlabel('Predicted probability')
    plt.ylabel('Frequency')


def dist_comparison(missing_values_dataframe, non_missing_values_dataframe):
    """
    credit - https://www.kaggle.com/kernels/scriptcontent/1415757/download
    
    """
    
    df1 = missing_values_dataframe
    df2 = non_missing_values_dataframe
    
    
    a = len(df1.columns)
    if a%2 != 0:
        a += 1
    
    n = np.floor(np.sqrt(a)).astype(np.int64)
    
    while a%n != 0:
        n -= 1
    
    m = (a/n).astype(np.int64)
    coords = list(itertools.product(list(range(m)), list(range(n))))
    
    numerics = df1.select_dtypes(include=[np.number]).columns
    cats = df1.select_dtypes(include=['category']).columns
    
    fig = plt.figure(figsize=(15, 20))
    axes = gs.GridSpec(m, n)
    axes.update(wspace=0.25, hspace=0.25)
    
    for i in range(len(numerics)):
        x, y = coords[i]
        ax = plt.subplot(axes[x, y])
        col = numerics[i]
        sns.kdeplot(df1[col].dropna(), ax=ax, label='datframe with Missing values').set(xlabel=col)
        sns.kdeplot(df2[col].dropna(), ax=ax, label='dataframe with NO Missing values')
        
    for i in range(0, len(cats)):
        x, y = coords[len(numerics)+i]
        ax = plt.subplot(axes[x, y])
        col = cats[i]

        df1_temp = df1[col].value_counts()
        df2_temp = df2[col].value_counts()
        df1_temp = pd.DataFrame({col: df1_temp.index, 'value': df1_temp/len(df1), 'Set': np.repeat('df1', len(df1_temp))})
        df2_temp = pd.DataFrame({col: df2_temp.index, 'value': df2_temp/len(df2), 'Set': np.repeat('df2', len(df2_temp))})

        sns.barplot(x=col, y='value', hue='Set', data=pd.concat([df1_temp, df2_temp]), ax=ax).set(ylabel='Percentage')
    
    return None


def encode_categorical_features(dataframe, strategy, list_of_features,list_of_features_to_skip=None):
    """
    This function will take a dataframe as input and perform the following:

    Encode the features passed as a list using the strategy selected. This
    function uses category_encoder library.

    For ordinal features, the functional will use ordinal_encoder
    For non-ordinal features, the function will use JamesStein Encoder

    :param dataframe: dataframe object
    :param strategy: this is the parameter that holds how the categorical features should be encoded
        strategy types available: 'ordinal', 'non_ordinal'
    :param list_of_features: pass a list object containing features to be encoded using strategy selected
    :param list_of_features_to_skip: pass a list object containing features to be omitted from encoding
    :return: dataframe with categorical features encoded.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")

    if strategy is None:
        raise ValueError("Please select a strategy to use")

    if list_of_features is None:
        raise ValueError("Please pass a list of features to be encoded")

    if not isinstance(list_of_features, list):
        raise ValueError("Object passed is not a dataframe")

    encoder_type = "james_stein"

    # split dataframe into features and target variable
    y = dataframe['loan_status']
    x = dataframe.drop("loan_status", axis=1)

    if strategy == 'ordinal':
        # create an ordinal encoder object
        ordinal_encoder = ce.OrdinalEncoder(cols=list_of_features)
        # transform the dataframe - returns a df with categorical features encoded
        dataframe = ordinal_encoder.fit_transform(x, y)

        # merge back the dataframe with the y target variable
        dataframe = dataframe.merge(y, on=y.index)
        # drop the index feature key_0
        dataframe.drop("key_0", axis=1, inplace=True)

        # convert the categorical features back to the category data type
        dataframe[list_of_features] = dataframe[list_of_features].astype('category')
    elif strategy == 'non_ordinal':
        # select all the features in the dataset
        non_ordinal = dataframe.select_dtypes(include='category').columns.tolist()
        # filter out non-ordinal features using ordinal features passed
        non_ordinal_features = [value for value in non_ordinal if value not in list_of_features]

        # create encoder object
        encoder = ce.JamesSteinEncoder(cols=non_ordinal_features)

        # transform the dataframe - returns a df with categorical features encoded
        dataframe = encoder.fit_transform(x, y)

        # merge back the target variable to the dataframe(df)
        dataframe = dataframe.merge(y, on=y.index)

        # drop the index feature key_0
        dataframe.drop("key_0", axis=1, inplace=True)

        # convert non_ordinal_features back to category data type
        dataframe[non_ordinal_features] = dataframe[non_ordinal_features].astype('category')

    return dataframe


def rfecv_feature_selection(df, target_feature, scoring, n_jobs, cross_val_count):
    """
    This function will take a dataframe as input and perform the following:

    Using Recursive Feature Selection method, it will select the top n
    features. We will also make sure to add fico_score and dti features
    if not selected in the final  list of features.

    :param df: pass dataframe name
    :param target_feature: pass target feature name
    :param scoring: target scoring metric
    :param cross_val_count: # of cross validation folds
    :param n_jobs: # of cpu jobs
    :return: dataframe with outliers removed.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")

    # drop our target feature
    x = df.drop(target_feature, axis=1)
    # save our target feature in a y variable
    y = df[target_feature]

    feature_names = x.columns
    import xgboost as xgb

    classifier = xgb.XGBClassifier(n_jobs=n_jobs,class_weight="balanced",max_depth=6,random_state=2019)
    #classifier = RandomForestClassifier(n_jobs=n_jobs,class_weight="balanced",max_depth=6,random_state=2019)
    cv = StratifiedKFold(n_splits=cross_val_count)

    rfecv_selector = RFECV(estimator=classifier,
                           step=1,
                           cv=cv,
                           scoring=scoring,
                           n_jobs=n_jobs)

    rfecv_selector.fit(x, y)

    print('Optimal number of features : %d' % rfecv_selector.n_features_)
    # print("RFE selected the {} # of features to be :".format(rfecv_selector.n_features_))

    # grab selected features
    top_n_features = feature_names[rfecv_selector.support_].tolist()

    df = x[top_n_features]
    df = df.iloc[:, 0:25]

    # merge back the target variable to the dataframe(df)
    df = df.merge(y, on=y.index)

    # drop generated index
    df.drop("key_0", axis=1, inplace=True)

    plt.figure(figsize=(11, 4))
    plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
    plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
    plt.plot(range(1, len(rfecv_selector.grid_scores_) + 1), rfecv_selector.grid_scores_, color='#303F9F',
             linewidth=3)

    plt.show()
    return df


def feature_selection_mlextend(dataframe, target_feature_name, scoring_metric, n_jobs, cross_val, range_of_features):
    """
    This function will take a dataframe as input and perform the following:

    a) Remove noise using Boruta Feature Selection
    b) Select the top features given a range of features to select

    :param dataframe: dataframe with features to select from
    :param target_feature_name: target name of the feature
    :param scoring_metric: f1_weighted, accuracy, roc_auc etc.
    :param cross_val: # of cross validation folds
    :param n_jobs: # of cpu jobs
    :param range_of_features: tuple of the range of features to select from (low, high)
    :return: dataframe with features reduced
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")

    if not isinstance(range_of_features, tuple):
        raise ValueError("range_of_features passed is not a tuple")

    import lightgbm as lgb

    classifier = lgb.LGBMClassifier(n_jobs=n_jobs, class_weight="balanced", max_depth=6, random_state=2019)

    x = dataframe.drop(target_feature_name, axis=1).values
    y = dataframe[target_feature_name].values.ravel()

    # forward selection

    sequential_forward_feature_selection = sfs(classifier,
                                               k_features=range_of_features,
                                               forward=True,
                                               n_jobs=n_jobs,
                                               floating=False,
                                               verbose=False,
                                               scoring=scoring_metric,
                                               cv=cross_val)

    sfs_algo = sequential_forward_feature_selection.fit(x, y)

    sfs_cross_val_score = sfs_algo.k_score_

    cross_val = round(sfs_cross_val_score, 2)

    selected_features = list(sfs_algo.k_feature_names_)

    print("Number of features selected is:  {}".format(len(sfs_algo.k_feature_names_)))

    print("Cross Validation Score for {}, is {}".format(scoring_metric, cross_val))

    plot_sfs(sfs_algo.get_metric_dict(), kind='std_err', figsize=(11, 7));

    df = x[selected_features]

    # merge back the target variable to the dataframe(df)
    df = df.merge(y, on=y.index)

    # drop generated index
    df.drop("key_0", axis=1, inplace=True)

    cat_features = []

    for col_name in df.columns:
        if df[col_name].dtype != 'float64':
            if col_name != 'loan_status':
                cat_features.append(col_name)

    print(cat_features)

    return df


def boruta_feature_selection(dataframe, target_feature_name, n_jobs):
    """
    This function will take a dataframe as input and perform the following:

    a) Remove noise using Boruta Feature Selection
    b) Select the top features given a range of features to select

    :param dataframe: dataframe with features to select from
    :param target_feature_name: target name of the feature
    :param n_jobs: # of cpu jobs
    :return: dataframe with features reduced
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")

    classifier = RandomForestClassifier(n_jobs=n_jobs, class_weight="balanced", max_depth=6, random_state=2019)

    x_temp = dataframe.drop(target_feature_name, axis=1)
    column_names = x_temp.columns
    y_temp = dataframe[target_feature_name]

    x = dataframe.drop(target_feature_name, axis=1).values
    y = dataframe[target_feature_name].values.ravel()

    """
    Boruta Py Code - credit: https://www.kaggle.com/rsmits/feature-selection-with-boruta
    """

    feat_selector = bpy.BorutaPy(classifier, n_estimators='auto', verbose=False, random_state=1)

    feat_selector.fit(x, y)

    filtered_dataframe = feat_selector.transform(x)

    final_features = list()

    indexes = np.where(feat_selector.support_ == True)

    for x in np.nditer(indexes):
        final_features.append(column_names[x])

    boruta_dataframe = x_temp[final_features]

    # merge back the target variable to the dataframe(df)
    df = boruta_dataframe.merge(y_temp, on=y_temp.index)

    # drop generated index
    df.drop("key_0", axis=1, inplace=True)

    cat_features = []

    for col_name in df.columns:
        if df[col_name].dtype != 'float64':
            if col_name != 'loan_status':
                cat_features.append(col_name)

    print("Categorical Features: {}".format(cat_features))

    return df


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.     
        https://www.kaggle.com/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def feature_selection(x_train, y_train, scoring, n_jobs, cross_val, classifier='rfc'):
    """
    This function will take a dataframe as input and perform the following:

    Using Recursive Feature Selection method, it will select the top n
    features. We will also make sure to add fico_score and dti features
    if not selected in the final  list of features.

    :param x_train: feature set
    :param y_train: target feature
    :param scoring: target scoring metric
    :param cross_val: # of cross validation folds
    :param n_jobs: # of cpu jobs
    :param top_n_features: The number of features to select (filter)
    :param classifier: Optional parameter to select classifier to use
    :return: dataframe with outliers removed.
    """
    
    from yellowbrick.model_selection import RFECV

    if not isinstance(x_train, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")

    x = x_train
    y = y_train

    feature_names = x.columns

    if classifier == 'rfc':
        classifier = RandomForestClassifier(n_jobs=n_jobs, class_weight="balanced", max_depth=6, random_state=2019)
    else:
        classifier = LogisticRegression(class_weight='balanced', n_jobs=-1, solver='lbfgs')

    rfecv_selector = RFECV(model=classifier, step=1, cv=cross_val, scoring=scoring, n_jobs=n_jobs)

    rfecv_selector.fit(x, y)

    print('Optimal number of features : %d' % rfecv_selector.n_features_)
    # print("RFE selected the {} # of features to be :".format(rfecv_selector.n_features_))
    
    rfecv_selector.show()
    
    # grab selected features
    top_n_features = feature_names[rfecv_selector.support_].tolist()

    x = x[top_n_features]

    return x


def remove_outliers(df, cpu_count, contamination='auto',):
    
    """
    This function will take a dataframe as input and perform the following:

    :param df: dataframe object
    :param  contamination: The amount of contamination of the data set,
            i.e. the proportion of outliers in the data set.
    :return: dataframe with outliers removed.
    """

    #if not isinstance(df, pd.DataFrame):
    #    raise ValueError("Object passed is not a dataframe")

    iso_forest_pipe = Pipeline([
        ('scale', StandardScaler()),
        ('isolation_forest', IsolationForest(n_jobs=cpu_count, contamination=contamination))
        ])

    iso_forest_pipe.fit(df)

    outlier_predictions = iso_forest_pipe.predict(df)

    df['outliers'] = outlier_predictions

    outliers = df.loc[df['outliers'] == -1].shape

    print("The number of outliers in the dataframe passed : {}".format(outliers))

    df = df[df['outliers'] != -1]

    df.drop('outliers', axis=1, inplace=True)

    return df


def impute_features(dataframe, strategy):
    """
    This function will take a dataframe as input and perform the following:

    Use dtypes to sub-select categorical features (uses == category data type)
    Impute the categorical features using given strategy.

    Use dtypes to sub-select numerical features (uses != category data type)
    Impute the numerical features using given strategy.

    :param dataframe: dataframe object
    :param  strategy: this is the parameter that hold the impute strategy
    :return: dataframe with categorical features imputed.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")

    if strategy == 'most_frequent':
        # create a list of categorical features by identifying them using the dtype ='category'
        categorical_list = dataframe.select_dtypes(include='category').columns.to_list()
        # use the Sklearn SimpleImputer
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        # fit transform our dataset
        dataframe[categorical_list] = categorical_imputer.fit_transform(dataframe[categorical_list])
        # convert categorical features back to the category data type as
        # simple imputer usually converts the data type to float32.
        # the step below converts them back to categorical
        dataframe[categorical_list] = dataframe[categorical_list].astype('category')
    elif strategy == 'median':
        # create a list of numerical features by identifying them using the dtype != category
        numerical_list = dataframe.select_dtypes(exclude='category').columns.to_list()
        # use the Sklearn SimpleImputer
        numerical_imputer = SimpleImputer(strategy='median')

        # fit transform our dataset
        dataframe[numerical_list] = numerical_imputer.fit_transform(dataframe[numerical_list])

    return dataframe


def drop_features_with_missing_values(dataframe, threshold_value=80):
    """
    This function will take a dataframe as input and perform the following:

    if method = 'threshold':
        drop features with missing values based on threshold

    :param dataframe: dataframe object containing features analyzed for missing values
    :param threshold_value: this value will be the cut-off used to determine if a feature
    is to be dropped. Features with missing values above (threshold_value) will be dropped.
    :param method: how to drop the features.
    :return: dataframe with features passed with missing features dropped
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")

    method = 'threshold'

    if method == 'threshold':
        # this first step transforms any present inf features to nan values
        # dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        # calculate the % missing in each column
        missing_values = (dataframe.isnull().sum() / dataframe.shape[0]) * 100
        # sort with highest values at the top
        missing_stats = pd.DataFrame(missing_values).rename(columns={'index': 'feature', 0: 'missing_percentage'}). \
            sort_values("missing_percentage", ascending=False)

        # find features with a missing % above a threshold
        missing_threshold = pd.DataFrame(missing_values[missing_values > threshold_value]).reset_index(). \
            rename(columns={'index': 'feature', 0: 'missing_percentage'}).sort_values("feature", ascending=True)

        # store features in a list to be used during the drop command
        missing_to_drop = list(missing_threshold['feature'])

        dataframe = dataframe.drop(labels=missing_to_drop, axis=1)

    return dataframe


def extract_numerical_features(dataframe, feature_names_to_extract, convert_to_categorical=None,):
    """
    This function will extract numeric features from a passed features
    that contain numerical values embedded in a string object.

    :param dataframe: dataframe object containing features to be extracted
    :param feature_names_to_extract: list of features to be extracted
    :param convert_to_categorical: features passed will be convert to categorical data type
    :return: dataframe with features passed converted to numeric features.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")

    if not isinstance(feature_names_to_extract, list):
        raise ValueError("Feature names passed should be a list object")

    # Extract numerical features
    for value in dataframe[feature_names_to_extract]:
        dataframe[value] = dataframe[value].str.extract('(\d+)').astype('float32')

    # have to refactor this. as of now, we convert these features to category because
    # the features passed for this value are category for this dataset.
    dataframe[convert_to_categorical] = dataframe[convert_to_categorical].astype('category')

    return dataframe


def impute_numerical_features(dataframe, strategy='constant', features_to_impute = None, fill_value=0):
    """
    This function will take as input a dataframe and drops
    any features within it that has single unique values.

    :param dataframe:
    :param strategy: default value = 'constant'
    :param fill_value: default value = 0
    :return: dataframe with
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Object passed is not a dataframe")

    if fill_value is None:
        raise ValueError("Missing the constant value to replace with")

    if features_to_impute is not None:
        # for the given dataframe, exclude category types and store in a list
        numerical_values = features_to_impute

        # use sklearn Simple imputer
        numerical_imputer = SimpleImputer(strategy='constant', fill_value=fill_value)

        # fit transform to dataframe
        dataframe[numerical_values] = numerical_imputer.fit_transform(dataframe[numerical_values])
    else:
        # for the given dataframe, exclude category types and store in a list
        numerical_values = dataframe.select_dtypes(exclude='category').columns.to_list()

        # use sklearn Simple Imputer
        numerical_imputer = SimpleImputer(strategy='constant', fill_value=fill_value)

        # fit transform to dataframe
        dataframe[numerical_values] = numerical_imputer.fit_transform(dataframe[numerical_values])

    print("{} features have been imputed with {} value".format(len(numerical_values), fill_value))

    return dataframe


def drop_low_variance_features(dataframe):
    """
    This function will take as input a dataframe and drops
    any features within it that has single unique values.

    :param dataframe:
    :return: dataframe with features dropped that have single unique values
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(" Object passed is not a dataframe")

    # store unique counts in each feature/column
    unique_counts = dataframe.nunique()

    # calculate unique counts in each column
    unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'}) \
        .sort_values('nunique', ascending=True)

    # find columns with only one unique count
    record_single_unique = pd.DataFrame(unique_counts[unique_counts < 2]).reset_index(). \
        rename(columns={'index': 'feature', 0: 'nunique'})

    # store features to drop in a list
    non_unique_to_drop = list(record_single_unique['feature'])

    print("The number of features with single unique values is {}".format(len(non_unique_to_drop)))

    # drop features
    dataframe.drop(labels=non_unique_to_drop, inplace=True, axis=1)

    return dataframe


def drop_highly_correlated_features(dataframe,threshold=0.90):
    """
    This function takes in a dataframe and calculates the correlation of the features.
    It will then drop features that cross the threshold passed.

    :param dataframe: takes the loan dataset dataframe
    :param threshold: takes in a threshold value. default = 0.90
    :return: dataframe with highly correlated features dropped
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(" Object passed is not a dataframe")

    y = dataframe['loan_status']
    x = dataframe.drop('loan_status', axis=1)

    # create a correlation matrix on the dataset
    corr = x.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))

    # find index of feature columns with correlation greater  than 90
    correlated_features = [column for column in upper.columns if any(upper[column] > threshold)]

    # drop correlated features
    dataframe = dataframe.drop(correlated_features, axis=1)

    print("Number of correlated features dropped = {}".format(len(correlated_features)))
    
    return dataframe


def encode_target_feature(dataframe):
    """
    This function takes the loan data set as input and converts the target feature as
    shown below:

    Class 1: Fully Paid Examples in the dataset
    Class 2: Charged off, late and defaulted examples.

    The target feature is also converted to a category data type

    :param dataframe: takes the loan dataset dataframe
    :return: loan dataset filtered for the two classes only
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(" Object passed is not a dataframe")

    # map loan status values to 1, 2 and 0
    dataframe['loan_status'] = dataframe['loan_status'].map({'Current': 2,
                                                             'Fully Paid': 1,
                                                             'Charged Off': 0,
                                                             'Late(31-120 days)': 0,
                                                             'In Grace Period': 0,
                                                             'Late(16-30 days)': 0,
                                                             'Does not meet the credit policy. Status:Fully Paid':0,
                                                             'Does not meet the credit policy. Status:Charged Off':0, 
                                                             'Default': 0}
                                                           )
    # we want to exclude current records as those are ongoing and haven't arrived to their final state
    dataframe = dataframe[dataframe.loan_status != 2]
    # apply a function to select only charged off and fully paid examples
    dataframe["loan_status"] = dataframe["loan_status"].apply(lambda loan_status: 0 if loan_status == 0 else 1)

    # convert our target feature to a category data type
    dataframe['loan_status'] = dataframe['loan_status'].astype('category')

    return dataframe


def convert_data_type(dataframe):
    """
    This function converts data types in a dataframe as shown below:
    object data types are converted to 'category' data type
    float64 data types are converted to 'float32' data type

    :param dataframe: receives as input a dataframe
    :return: a dataframe with the conversions mentioned above.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(" Object passed is not a dataframe")

    for value in dataframe:
        if dataframe[value].dtype == 'object':
            dataframe[value] = dataframe[value].astype('category')

    return dataframe


def missing_values_table(dataframe):
    """
    This function will calculate missing values for each feature in a given
    dataframe.

    :param dataframe:
    :return:
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError(" Object passed is not a dataframe")

    mis_val = dataframe.isnull().sum()
    mis_val_percent = 100 * dataframe.isnull().sum() / len(dataframe)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(dataframe.shape[1]) + " columns.\n"
                                                                     "There are " + str(
        mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns


def data_dictionary(data=None):
    """
    This helper function helps us get review field defintions
    given a list of field
    names
    """

    # lets load our data dictionary to review some column descriptions
    dd = pd.read_excel('LCDataDictionary.xlsx')
    # Lets increase the width of the columns to clearly see the descriptions
    pd.set_option('max_colwidth', 1200)

    dictionary_df = pd.DataFrame(dd.loc[dd["feature"].isin(data)]). \
        drop_duplicates().sort_values("feature", ascending=True)
    return dictionary_df


def yellow_brick_learning_curve(model, x, y, cpu_count, cv_count, scoring_metric):
    """

    """
    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=cv_count)
    sizes = np.linspace(0.3, 1.0, 10)

    # Instantiate the classification model and visualizer
    visualizer = LearningCurve(
        model, cv=cv, scoring=scoring_metric, train_sizes=sizes, n_jobs=cpu_count)

    visualizer.fit(x, y)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure


def yellow_brick_validation_curve(model, x, y, cpu_count, cv_count, param, scoring_metric):
    """

    """
    from yellowbrick.model_selection import LearningCurve
    from sklearn.model_selection import StratifiedKFold

    # Create the learning curve visualizer
    cv = StratifiedKFold(n_splits=cv_count)
    # Validation Curve

    mpl.rcParams['axes.prop_cycle'] = cycler('color', ['purple', 'darkblue'])

    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    ax = plt.subplot(411)

    viz = ValidationCurve(model,
                          n_jobs=cpu_count,
                          ax=ax,
                          param_name=param,
                          param_range=np.arange(1, 11),
                          cv=cv,
                          scoring=scoring_metric)

    # Fit and poof the visualizer
    viz.fit(x, y)
    viz.show();


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
  
    return plt


