# Rob's functions for use in kaggle tabular data projects
# Please let me know if you use !!!
# Feedback always appreciated

####################
# Import common libraries and toolkits 
import numpy as np
import pandas as pd 

import sklearn as skl
import lightgbm as lgb
import xgboost as xgb
import catboost as catb

import torch

import math
from scipy import stats
import random
import itertools

import optuna
from tqdm import tqdm
from time import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as go

from multiprocessing import cpu_count

#import plotly.express as px
#from statsmodels import graphics
#from pyexpat import features
#import statsmodels.api as sm
#import statsmodels.formula.api as smf


###Initiaalization and data loading functions
def set_globals(seed: int = 67, verbose: bool=True):
    """
    -----------
    sets: global variables and configurations for the project
    - seed settings for reproducibility
    - pandas display options
    - seaborn/matplotlib visualization styles
    -----------
    returns:
    - DEVICE: torch device (cpu or cuda)
    - CORES: number of CPU cores to use for multiprocessing
    -----------
    requires: numpy, pandas, seaborn, matplotlib, random, multiprocessing.cpu_count
    optional: torch
    """
    # random seed settings for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # visualization settings
    pd.set_option('display.max_rows', 25)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.width', 130)
    pd.set_option('display.precision', 4)

    # custom seaborn/matplotlib style
    MY_PALETTE = get_colors()
    sns.set_theme(context = 'paper', style = 'ticks', palette = MY_PALETTE, rc={"figure.figsize": (9, 3), "axes.spines.right": False, "axes.spines.top": False})

    # device settings for torch
    try: 
        torch.manual_seed(seed)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    except: 
        DEVICE = 'cpu'
    CORES = min(4, cpu_count())  # Limit cores to avoid memory issues

    if verbose:
        print(f"Using device: {DEVICE}")
        print(f"Using {CORES} CPU cores when multiprocessing")
    
    return DEVICE, CORES 

def get_colors(color_keys: list=None, get_cmap: bool=False, 
                cmap_name: str='cividis', n_hues: int=1, n_sats: int=1):
    """
    generates a palette or color map for visualizations
    -----------
    returns:
    if n_colors is provided:
    - generates and returns a matplotlib colormap object or list of colors with n_colors
    else:
    - default colormap object or list of colors for visualizations
    -----------
    requires: seaborn, matplotlib
    """
    MY_PALETTE = sns.xkcd_palette(['ocean blue', 'gold', 'dull green', 'dusty rose', 'dark lavender', 'carolina blue', 'sunflower', 'lichen', 'blush pink', 'dusty lavender'])
    if color_keys is None:
        if get_cmap: return mpl.colormaps[cmap_name]
        else: return MY_PALETTE

    n_colors = len(color_keys)
    if get_cmap:
        if n_colors <= len(MY_PALETTE):
            return dict(zip(color_keys, MY_PALETTE[:n_colors]))
        elif n_colors <= n_hues * n_sats:
            new_palette = []
            for j in range(n_hues):
                for i in range(n_sats):
                    new_palette.append(sns.desaturate(MY_PALETTE[j], 1-.2*i))
            return dict(zip(color_keys, new_palette[:n_colors]))
        else:
            cmap = mpl.colormaps[cmap_name].resampled(n_colors)
            return dict(zip(color_keys, [cmap(i / n_colors) for i in range(n_colors)]))
    else:
        #if n_colors == 1: return sns.xkcd_palette(['steel']) 
        if n_colors <= len(MY_PALETTE):
            return MY_PALETTE[:n_colors]
        elif n_colors <= n_hues * n_sats:
            new_palette = []
            for j in range(n_hues):
                for i in range(n_sats):
                    new_palette.append(sns.desaturate(MY_PALETTE[j], 1-.2*i))
            return new_palette[:n_colors]
        else:
            cmap = mpl.colormaps[cmap_name].resampled(n_colors)
            return [cmap(i / n_colors) for i in range(n_colors)]       

def summarize_data(df_: pd.DataFrame, features: list)-> None:
    """prints df summary and descriptive stats for selected features"""
    try:
        df = df_[df_.target_mask.eq(True)][features]
    except:
        df = df_[features]
    print("=" * 69)
    print(df_[features].info())
    print("=" * 69)
    print(df.head(5).T) 
    print("=" * 69)
    try:
        print(df.describe(include=['float', 'int']).T)
    except: pass
    try:
        non_numeric_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
        print(df[non_numeric_cols].describe().T)
    except: pass
    
def load_tabular_data(path: str, extra_data: str=None, verbose: bool=True, csv_sep: str=","):
    """
    loads Kaggle type tabular data from path into single DataFrame
    -----------
    returns:
    - merged DataFrame for EDA & feature engineering
    - list of training features
        - cleans for consistent pandas friendly feature names
    - list of targets, including column "target_mask"
        - adds "target_mask" for separating test from training data
    - target column name (first target)
    -----------
    assumes:
    - path contains train.csv, test.csv, sample_submission.csv files
    - extra_data is optional "path + file name" of additional training data
        - extra_data contains same features and targets as train.csv
            - may need to preprocess extra data
    requires: pandas
    """
    df_train = pd.read_csv(f"{path}/train.csv", sep=csv_sep)
    df_test = pd.read_csv(f"{path}/test.csv", sep=csv_sep)
    df_submission = pd.read_csv(f"{path}/sample_submission.csv", sep=csv_sep)
    
    targets = list(df_submission.columns)
    features = list(df_test.columns)
    id_feature = [feature for feature in features if feature in targets]
    assert len(id_feature) == 1, "Expected exactly one ID column"
    targets = [feature for feature in targets if feature not in id_feature]
    features = [feature for feature in features if feature not in id_feature]
   
    df_test = df_test.merge(df_submission, how = 'left', on = id_feature)
    df = pd.concat([df_train.assign(target_mask = True), df_test.assign(target_mask = False)], ignore_index=True)
    
    if extra_data != None:
        #TODO: validate extra_data loading
        df_extra_training = pd.read_csv(f"{extra_data}", sep=csv_sep)
        missing = set(targets + features + id_feature) - set(df_extra_training.columns)
        assert not missing, f"Extra Data missing columns: {missing}"
        df_extra_training[id_feature[0]] = range(len(df), len(df) + len(df_extra_training))
        df = pd.concat([df, df_extra_training.assign(target_mask = True)])
    
    df.set_index(id_feature, inplace = True)  
    ### clean feature names
    clean_feature_names = {}
    for i, col in enumerate(features):
        clean_feature_names[col] = col.casefold().strip().replace(" ","_").replace("(","_").replace(")","").replace("-","_").replace(".","_")
        features[i] = clean_feature_names[col]
    df.rename(columns=clean_feature_names, inplace=True)

    if verbose:
        print("=" * 69)
        print(f"Loaded {df.target_mask.eq(True).sum()} training samples of {len(features)} predictive features and {len(targets)} target(s) in DataFrame.")
        print(f"Loaded {df.target_mask.eq(False).sum()} testing samples in DataFrame.")
        print(f"DataFrame shape: {df.shape}. Ready to explore, engineer, and predict!")
        print("=" * 69)
    targets.append('target_mask')

    return df, features, targets, targets[0]

def get_target_labels(df: pd.DataFrame, target: str, targets: list, cuts: int=8, verbose: bool=True):
    """
    Adds target "label" columns
    Useful for visualizing numeric targets in categorical "bins"
    -----------
    if target is numeric with more unique values than cuts
        - adds "qcut_label" using pd.qcut to create quantile-based bins of the target
        - adds "label" using pd.cut to create equal-width bins of the target
    returns:
    - df with new label columns added
    """
    if (df[target].dtype == int or df[target].dtype == float) and df[target].nunique() > cuts:
        df["qcut_label"] = pd.qcut(df[df.target_mask.eq(True)][target], cuts, labels=False)
        df["label"] = pd.cut(df[df.target_mask.eq(True)][target], cuts, labels=False)
        df[["qcut_label", "label"]] = df[["qcut_label", "label"]].fillna(-1).astype('int16').astype('category')
        if "qcut_label" not in targets: targets.append("qcut_label")
        if "label" not in targets: targets.append("label")
        if verbose:
            print(f'Added label and qcut_label as categorical targets')
        return df, targets, "label"
    else:
        print(f'NOTE: target is label is {target}')
        return df, targets, target

###Data cleaning and feature engineering functions
def check_duplicates(df: pd.DataFrame, features: list, target: str, drop: bool=False, reset_index: bool=False, verbose: bool=True) -> pd.DataFrame:
    """
    Checks for duplicate rows of selected features in df.
    Returns:
    - DataFrame with duplicate rows dropped from training data if drop=True.
    Requires: pandas
    """
    try:
        mask = df.target_mask.eq(True)
        train_df = df[mask]
        test_df = df[~mask]
    except:
        print("Unable to separate test and training data")
        print(f"{df[features].duplicated(keep=False).sum()} duplicates in DataFrame")
        return

    # Find duplicates in training data
    duplicate_mask = train_df[features].duplicated(keep=False)
    n_duplicates = duplicate_mask.sum()

    # Find overlap between train and test
    overlap = pd.merge(train_df[features], test_df[features], how='inner')
    n_overlap = overlap.shape[0]

    print("=" * 69)
    print(f"There are {n_duplicates} duplicated rows in the training data set.")
    print(f"There are {n_overlap} overlapping observations that appear in both the train and test data sets")
    print("=" * 69)

    if verbose and n_duplicates > 0:
        print(train_df[duplicate_mask][features].head(10).T)
        print("=" * 69)
        plot_target_eda(train_df[duplicate_mask], target, title="target distribution by duplicated rows")
        print("=" * 69)
        #TODO: add more visualizations comparing duplicated rows to non-duplicated rows in training data and to test data
        #TODO: show examples of duplicated rows, highlight differences in target values for duplicated rows, and visualize feature distributions for duplicated vs non-duplicated rows to look for patterns that could explain the duplicates or target differences
        #TODO: for duplicates in training data, compare how different the target values are for duplicated rows and if there are any patterns in the features that could explain the duplicates or target differences
    if drop and n_duplicates > 0:
        print("Dropping duplicated rows from training data set...")
        df.loc[mask, :] = train_df.drop_duplicates(subset=features, keep="last")
        df = df.dropna(subset=features)
        #TODO: only reset index on the training dataset. can we do with df.loc?
        if reset_index: df = df.reset_index(drop=True)
        train_df = df[mask] #reset train_df after dropping rows 
        duplicate_mask = train_df[features].duplicated(keep=False)
        overlap = pd.merge(train_df[features], test_df[features], how='inner')
        print(f"{duplicate_mask.sum()} duplicated rows remain in training data set.")
        print(f"{overlap.shape[0]} overlapping observations remain in the train and test data sets")
        if reset_index: print("Index reset.")
        else: print("Index NOT reset")
        print("=" * 69)
    return df

def plot_null_data(df:pd.DataFrame, features:list, verbose:bool=True)->list:
    def _calculate_null(df, features=features):
        ds = (df[features].isnull().sum() / len(df)) * 100
        return ds

    def _plot_null(ds, title="percentage of missing values in training data"):
        ds.sort_values(ascending=False, inplace = True)
        ds = ds[ds > 0]
        ds[:10].plot(kind = 'barh', title = f"Top {min(10, len(ds))} of {len(ds)} Features")
        plt.title(title)
        plt.xlabel('Percentage')
        plt.show()
    def _plot_missing_heatmap(df, title="null value heatmap"):
        sample_size = min(df.shape[0], 1000)
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.sample(sample_size).isnull(), cbar=False, cmap='cividis')
        plt.title(title)
        plt.show()
        
    ds_train = _calculate_null(df[df.target_mask.eq(True)])
    ds_test = _calculate_null(df[df.target_mask.eq(False)])
    if (ds_train == 0).all() and (ds_test == 0).all():
        print("=" * 69)
        print("No Missing Data")
        print("=" * 69)
        return []
    else:
        print("=" * 69)
        print("                    Percent Null")
        print("=" * 69)
        df_display = ds_train.to_frame(name = 'training data')
        df_display['test data']  = ds_test
        df_display = df_display.loc[~(df_display == 0).all(axis=1)]
        df_display.round(4)
        print(df_display)
        if verbose:
            _plot_null(ds_train)
            _plot_missing_heatmap(df)
        return df_display.index.tolist()

def plot_feature_transforms(df_: pd.DataFrame, feature: str)-> None:
    """
    Plots histogram distribution of feature variable with different transformations 
    Informs feature transformation feature engineering decisions 
    -----------
    Assumes feature is numeric and has many unique values (not categorical or boolean or numeric with few unique values)
    -----------
    requires: pandas, numpy, seaborn, matplotlib, scikit learn
    """
    try:
        df = df_[df_.target_mask.eq(True)][feature].to_frame()
    except:
        df = df_[feature].to_frame()
    X = df[feature].values.reshape(-1,1)
    df['StandardScaler']  = skl.preprocessing.StandardScaler().fit_transform(X)
    df['PowerTransformer']  = skl.preprocessing.PowerTransformer().fit_transform(X)
    df_plot['QuantileTransformer']  = skl.preprocessing.QuantileTransformer().fit_transform(X)
    df['MinMaxScaler'] = skl.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    df['y_logTransform'] = np.log1p(X)
    columns = list(df.columns)
    fig, axs = plt.subplots(nrows=1, ncols=len(columns), sharey=True, figsize=(15,3))
    for i, col in enumerate(columns):
        plt.subplot(1, len(columns), i+1)
        sns.histplot(data=df, stat='percent', x=col, kde=False, bins=30, legend = False)
    plt.show()

def get_target_transformer(df: pd.DataFrame, target: str, 
                           targets: list, name: str="enc",
                           TargetTransformer=skl.preprocessing.StandardScaler(), 
                           verbose: bool=True
                           ):
    """
    scales or transforms targets in df with scikit learn scalers / transformers
    -----------
    returns:
    - df with transformed target
    - updated list of targets with new transformed target column name added
    - fitted TargetTransformer to support inverse transformation of predictions
    -----------
    requires: 
    pandas, scikit learn
    """
    t=target.casefold().strip().replace(" ","_").replace("(","_").replace(")","").replace("-","_").replace(".","_")
    enc_tgt = f"{t}_{name}"
    df[enc_tgt] = -1

    mask = df.get("target_mask", pd.Series(True, index=df.index))
    y_fit = df.loc[mask, target].values.reshape(-1, 1)
    y_trans = TargetTransformer.fit_transform(y_fit).ravel()
    df.loc[mask, enc_tgt] = y_trans

    targets = targets + [enc_tgt]

    if verbose:
        print(f"Added transformed target '{t}_{name}' to DataFrame")
        plot_target_eda(df, enc_tgt, title=f"Distribution of Transformed Target '{t}_{name}'")
    return df, targets, TargetTransformer

def clean_categoricals(df: pd.DataFrame, features: list, 
                       string_length: int=3, fillna:bool=True) -> pd.DataFrame:
    """
    edits strings in categorical feature columns to be more consistent and easier to work with
    -----------
    returns: 
    updated df with cleaned categorical feature columns
        - converted to lowercase
        - replaces spaces and special characters
        - trims to specified max length
        - converts to category dtype
    -----------
    requires: pandas
    """
    for col in df[features].select_dtypes(include=['object', 'string']).columns:
        if fillna: 
            df[col].fillna('unk', inplace=True) 
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.casefold()
            .str.replace("-", "_", regex=False)
            .str.replace(". ", "__", regex=False)
            .str.replace(", ", "_", regex=False)
            .str.replace(" ", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[col] = df[col].str[:string_length].astype('category')
    return df

def split_training_data(df: pd.DataFrame, features: list, targets, 
                        drop_na: bool=False, validation_size: None| float | pd.Index = None):
    """
    splits df into training, validation, and test sets
    -----------
    if targets is a list, y will be a DataFrame, if targets is a string, y will be a Series
    *** if y is a DataFrame it may need to be converted to Series for some models e.g. y = y[target] ***
    if validation_size is None,
        returns train and test splits of the data
        use to only separate test from training data without creating a validation set
    if validation_size is float (0-1),
        returns train, validation, and test splits of the data based on percentage of training data
        use to create a random validation set from the training data
    if validation_size is pd.Index,
        returns train, validation, and test splits of the data based on selected rows in the training data
        use to create a specific validation set e.g. cross validation fold or time-based split
    -----------
    requires: pandas, scikit learn
    """
    def _drop_nan(X, y):
        nan_idx = X[X.isna().any(axis=1)].index
        X = X.drop(nan_idx)
        y = y.drop(nan_idx)
        return X, y

    SEED = 80085
    X_test = df[df.target_mask.eq(False)][features]
    y_test = df[df.target_mask.eq(False)][targets]

    X = df[df.target_mask.eq(True)][features]
    y = df[df.target_mask.eq(True)][targets]
    
    if drop_na:
        X, y = _drop_nan(X, y)
    if validation_size is None:
        return X, y, X_test, y_test, X_test, y_test
    elif type(validation_size) is float:
        X_train, X_val, y_train, y_val  = skl.model_selection.train_test_split(X, y, test_size = validation_size, random_state = SEED)
        return X_train, y_train, X_val,  y_val, X_test, y_test
    elif type(validation_size) is pd.Index: 
        X_train, y_train = X[~X.index.isin(validation_size)], y[~y.index.isin(validation_size)]
        X_val, y_val = X[X.index.isin(validation_size)], y[y.index.isin(validation_size)]
        return X_train, y_train, X_val,  y_val, X_test, y_test
    else: return X, y, X_test, y_test, X_test, y_test

def get_transformed_features(df: pd.DataFrame, features: list, FeatureTransformer, winsorize: list=[0,0]):
    """
    scales or transforms features in df with scikit learn scalers / transformers
    -----------
    returns: df with transformed features
    -----------
    requires: pandas, scikit learn, tqdm
    """
    for feature in tqdm(features, desc="Transforming features", unit="features"):
        if winsorize != [0,0]:
            stats.mstats.winsorize(df[feature], limits=winsorize, inplace=True)
        X = df[feature].values.reshape(-1,1)
        if FeatureTransformer is not None:
            df[feature] = FeatureTransformer.fit_transform(X)
    return df    

def get_feature_interactions(df: pd.DataFrame, features:list, winsorize: list=[0,0], transform: bool=True) -> pd.DataFrame:
    """
    adds interaction features to df 
    ----------
    returns: df with new interaction features added
        - multiplys pairs of features together 
        - applies a power transformation to the new features
    ----------- 
    requires: pandas, scikit learn, itertools, tqdm
    """
    ##Add kitchen sink feature inteactions 
    for combination in tqdm(itertools.combinations(features, 2), desc="Creating interaction features", unit="pairs"):
        df["*".join(combination)] = df[list(combination)].prod(axis=1)
    new_features = ["*".join(c) for c in itertools.combinations(features, 2)]
    if transform:
        df = get_transformed_features(df, new_features, skl.preprocessing.PowerTransformer(), winsorize=winsorize)
    print(f"Added {len(new_features)} inteaction features")
    return df

def get_feature_by_grouping_on_cat(df:pd.DataFrame, categorys:list, target:str,)->pd.DataFrame:
    """
    predicts the value and variance of a numeric target feature when grouped on a category
    -------
    requires: pandas
    """
    for category in categorys:
        group_stats = df[df.target_mask.eq(True)].groupby(category)[target].agg(['mean', 'std']).rename(
            columns={'mean': f'{target}_by_{category}_m', 'std': f'{target}_by_{category}_std'})
        df = df.merge(group_stats, on=category, how='left')
    return df

def get_feature_cat_interactions(df: pd.DataFrame, features:list, pivot:str)-> pd.DataFrame:
    """
    returns each combination of category values as a new category
    --------
    remember to check categories after
    --------
    requires: pandas, scikit learn, numpy
    """
    if df[pivot].nunique() > 16:
        print(f"{pivot} has too many values: {df[pivot].nunique()}")
        return df
    for feature in features:
        if df[feature].nunique() < 16:
            df[f'{feature}_on_{pivot}'] = df[feature].astype(str) + df[pivot].astype(str)
            X = df[f'{feature}_on_{pivot}'].values
            df[f'{feature}_on_{pivot}'] = skl.preprocessing.OrdinalEncoder().fit_transform(X.reshape(-1, 1))
            df[f'{feature}_on_{pivot}'] = df[f'{feature}_on_{pivot}'].astype('category')
    return df

def get_embeddings(df: pd.DataFrame, features:list, mapper, col_names:str, sample_size: float=None, target:str=None, index: int=1, verbose=True) -> pd.DataFrame:
    """
    fits a mapper to a sample of the data and then applys the mapping function to the full dataset to create new features
    useful for generating UMAP encoding, PCA loadings or kernal approximations of selected feature space    
    -----------
    returns: 
    df with new features added for the encoding
    if sample_size is provided, fits the mapper to a sample of the data and applies the mapping function to the full dataset
    if sample_size is not provided, fits the mapper to the full dataset and uses the fitted model to transform the data
    -----------
    assumes: mapper is a scikit learn PCA or kernel approximation object with fit_transform method or a UMAP object with fit and transform methods
    requires: numpy, pandas, time, matplotlib, seaborn
    optional: scikit learn, umap
    """
    tic=time()
    print("Training embedding function...")
    if sample_size is not None:
        if sample_size < 1.0:
            n = min(int(df[df.target_mask.eq(True)].shape[0] * sample_size), 10000)
        else:
            n = min(int(df[df.target_mask.eq(True)].shape[0]), int(sample_size))
        df_sample = df[df.target_mask.eq(True)].sample(n=n, random_state=69)
        mapper = mapper.fit(df_sample[features])
        print("Mapping features to embeddings...")
        reduced_data = mapper.transform(df[features])
    else:
        try:
            reduced_data = mapper.fit_transform(np.float32(df[features]))
        except:
            mapper = mapper.fit(df[features])
            print("Mapping features to embeddings...")
            reduced_data = mapper.transform(df[features])
    cols = [(col_names + str(i)) for i in range(index, index + reduced_data.shape[1])]
    X_features = pd.DataFrame(reduced_data, columns=cols, index=df.index)
    print(f"Added {len(cols)} {col_names} embedding features in {time()-tic:.2f}sec")

    if verbose:
        fig, ax = plt.subplots(figsize=(5, 3))
        if target != None:
            palette = get_colors(df[target].unique(), get_cmap=True)
            X_features[target] = df[target]
            hue=target
        else: 
            palette = get_colors()
            hue=None
        df_sampled = X_features.sample(n=min(800, X_features.shape[0]), random_state=69)
        sns.scatterplot(data=df_sampled, x=cols[0], y=cols[1], hue=hue, 
                        ax=ax, legend=False, palette=palette
                       ).set_title(f"{cols[1]} vs {cols[0]}")
        plt.xticks(())
        plt.yticks(())
        plt.xlabel("")
        plt.ylabel("")
        if target != None: X_features.drop(target, inplace = True, axis = 1)
        plt.show()
    return df.join(X_features)

def get_pls_embeddings(df: pd.DataFrame, features:list, target:list, col_names:str, 
                   sample_size: float=None, n_components:int=2, verbose=True) -> pd.DataFrame:
    """
    uses PLS Regression as a supervised dimension reduction 
    -----------
    returns: 
    df with new features added for the encoding
    if sample_size is provided, fits the mapper to a sample of the data and applies the mapping function to the full dataset
    if sample_size is not provided, fits the mapper to the full dataset and uses the fitted model to transform the data
    -----------
    assumes: mapper is a scikit learn PCA or kernel approximation object with fit_transform method or a UMAP object with fit and transform methods
    requires: numpy, pandas, time, matplotlib, seaborn, scikit
    """
    tic=time()
    print("Training PLS embedding function...")
    mapper = skl.cross_decomposition.PLSRegression(n_components=n_components)
    if sample_size is None:
        df_train = df[df.target_mask.eq(True)]    
    elif sample_size <= 1:
        n = min(int(df_train.shape[0] * sample_size), 10000)
        df_train = df[df.target_mask.eq(True)].sample(n=n, random_state=69)
    elif sample_size > 1:
        n = min(df_train.shape[0], int(sample_size))
        df_train = df[df.target_mask.eq(True)].sample(n=n, random_state=69)
    mapper = mapper.fit(df_train[features], df_train[target])
    print("Mapping features to embeddings...")
    reduced_data = mapper.transform(df[features])

    cols = [(col_names + str(i)) for i in range(1, 1 + reduced_data.shape[1])]
    X_features = pd.DataFrame(reduced_data, columns=cols, index=df.index)
    print(f"Added {len(cols)} {col_names} embedding features in {time()-tic:.2f}sec")

    if verbose:
        fig, ax = plt.subplots(figsize=(5, 3))
        if target != None:
            palette = get_colors(df[target].unique(), get_cmap=True)
            X_features[target] = df[target]
            hue=target
        else: 
            palette = get_colors()
            hue=None
        df_sampled = X_features.sample(n=min(800, X_features.shape[0]), random_state=69)
        sns.scatterplot(data=df_sampled, x=cols[0], y=cols[1], hue=hue, 
                        ax=ax, legend=False, palette=palette).set_title(f"{cols[1]} vs {cols[0]}")
        plt.xticks(())
        plt.yticks(())
        plt.xlabel("")
        plt.ylabel("")
        if target != None: X_features.drop(target, inplace = True, axis = 1)
        plt.show()
    return df.join(X_features)

def get_clusters(df: pd.DataFrame, features:list, encoder, col_name:str, target:str=None, verbose:bool=True) -> pd.DataFrame:
    """
    generates clusters for selected feature space
    -----------
    returns: updated df with new column for cluster labels
    - if target is provided, replaces cluster labels with mean target value creating a target-informed cluster feature
    -----------
    assumes: encoder is a scikit learn clustering object with fit_predict method
    requires: numpy, pandas, scikit learn, time, matplotlib, seaborn
    """
    tic = time()
    X = df[features].values
    if verbose: print(f"Encoding cluster feature '{col_name}'")
    df[col_name] = encoder.fit_predict(X)
    df[f"{col_name}_noise"] = df[col_name] == -1
    if target is not None:
        ds = df[df.target_mask.eq(True)].groupby(col_name)[target].mean()
        d = ds.to_dict()
        df[col_name].replace(d, inplace=True)
        if verbose: print(f"Cluster feature '{col_name}' replaced with mean target value by cluster")
    else:
        df[col_name] = df[col_name].astype('category')
    if verbose:
        print(f"Added cluster feature '{col_name}' with {df[col_name].nunique()} unique values in {time()-tic:.2f}sec")
        noise_pct = 100 * df[f"{col_name}_noise"].sum() / df.shape[0]
        if noise_pct > 0: print(f"Cluster feature '{col_name}' identified {noise_pct:.2f}% noise")
        palette = get_colors(color_keys=df[col_name].unique(), get_cmap=True)
        if target is not None:
            try:
                plot_features_eda(df[df.target_mask.eq(True)], [col_name], target, label=None)
            except:
                plot_features_eda(df, [col_name], target, label=None)
        df_sampled = df[df[f"{col_name}_noise"]==False].sample(n=min(800, df.shape[0]), random_state=69)
        reduced_data = skl.decomposition.PCA(n_components=2).fit_transform(df_sampled[features])
        df_sampled['pca_x'] = reduced_data[:,0]
        df_sampled['pca_y'] = reduced_data[:,1]
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.scatterplot(data=df_sampled, x=df_sampled['pca_x'], y=df_sampled['pca_y'], 
                            hue=col_name, alpha = 0.7, palette=palette, ax=ax, legend=False)
        plt.xticks(())
        plt.yticks(())
        plt.xlabel("")
        plt.ylabel("")
        plt.show()
    if not df[f"{col_name}_noise"].any():
        df.drop(columns=f"{col_name}_noise", inplace=True)
    return df

def denoise_categoricals(df: pd.DataFrame, features: list, target: str=None, threshold:float=0.1)-> pd.DataFrame:
    """
    identifies and removes noise from categorical features based on unique value counts and target distribution patterns
    -----------
    returns: df with 
        - new de-noised categorical features added for features where noise above threshold was identified
        - new target mean features when target is provided
    -----------
    requires: pandas, numpy
    """
    def _get_mean_std(df, df_train, feature, target):
        tgt_mean=df_train.groupby(feature)[target].mean().to_dict()
        tgt_std=df_train.groupby(feature)[target].std().to_dict()
        df[f"{feature}_tgt_mu"] = df[feature].replace(tgt_mean).astype('float32')
        df[f"{feature}_tgt_std"] = df[feature].replace(tgt_std).astype('float32')
        return df
    
    df[features] = df[features].astype('category')
    df_train = df[df.target_mask.eq(True)]
    df_test = df[df.target_mask.eq(False)]
    #assume threshold is given in percent
    noise_ceil_train = int(0.01 * threshold * df_train.shape[0])
    noise_ceil_test = int(0.01 * threshold * df_test.shape[0])

    for feature in features:
        noise = -1 if df[feature].cat.categories.dtype == 'int' or df[feature].cat.categories.dtype == 'float' else "noise"
        train_v = df_train[feature].unique()
        test_v = df_test[feature].unique()
        train_noise = [f for f in train_v if f not in test_v]
        test_noise = [f for f in test_v if f not in train_v]
        values = [f for f in train_v] + test_noise
        if len(train_v) < 2:
            print(f"{feature} is trivial, dropping {feature}")
            df.drop(col=feature, inplace=True)
        else:
            noise_dict = {}
            for v in values:
                if v in test_noise:
                    noise_dict[v] = noise
                elif v in train_noise:
                    noise_dict[v] = noise
                elif df_train.groupby(feature)[target].count()[v] < noise_ceil_train:
                    noise_dict[v] = noise
                elif df_test.groupby(feature)[target].count()[v] < noise_ceil_test:
                    noise_dict[v] = noise
            if len(noise_dict.keys()) == 0:
                print(f"No noise identified in {feature}")
                if target is not None:
                    df = _get_mean_std(df, df_train, feature, target)
            elif len(noise_dict.keys()) > 1:
                df_train[f"{feature}_denoise"] = df_train[feature].replace(noise_dict)
                training_noise = df_train[df_train[f"{feature}_denoise"].eq(noise)].shape[0]
                if  training_noise > 0:
                    df[f"{feature}_denoise"] = df[feature].replace(noise_dict).astype('category')
                    print(f"✔️ successfully de-noised {feature}: {100*training_noise/df_train.shape[0]:.2f}% noise in training data")
                    if target is not None:
                        df = _get_mean_std(df, df_train, feature, target)
                else:
                    print(f"""❌ Unable to denoise {feature}.\n
                              training noise: {training_noise} samples\n
                              test noise: {df_test[df_test[f"{feature}_denoise"].eq(noise)].shape[0]} samples
                          """)
            else:
                print(f"❌ Unable to denoise {feature}. Only one noise feature noise: {noise_dict.keys()}")
                if target is not None:
                    df = _get_mean_std(df, df_train, feature, target)
    return df  

def impute_using(df: pd.DataFrame, impute_informant: str, impute_features:list)-> pd.DataFrame:
    """
    uses a control variable to impute values into a list of features with missing data
    rather than simply filling with overall mean, fills with mean based on quantile cuts of informant feature
    ------- 
    requires: scikit-learn, pandas
    """
    df['control'] = skl.preprocessing.QuantileTransformer(n_quantiles=1000).fit_transform(
        df[[impute_informant]]
    )
    for f in impute_features:
        ds = df.groupby('control')[f].transform(lambda g: g.fillna(g.mean()))
        df[f] = ds
        df[f].fillna(df[f].mean(), inplace=True)
    df.drop('control', axis=1, inplace=True)
    return df

def get_outliers(df: pd.DataFrame, feature: str, deviations: int=4, 
                 remove: bool=False, verbose: bool=False) -> pd.DataFrame:
    """
    identifies outliers in a specified feature of a DataFrame
    -----------
    returns: DataFrame with outlier rows optionally removes, and DataFrame of outliers
    -----------
    requires: pandas, numpy
    """
    df_train = df[df.target_mask.eq(True)]
    m = df_train[feature].mean()
    s = df_train[feature].std()
    pop = df_train.shape[0]
    print("=" * 69)
    print(f"{pop} samples with mean: {m:.4f} std: {s:.4f}")
    for i in range(1, deviations+1):
        df_outlier = df_train[(df_train[feature] > i*(m+s)) | (df_train[feature] < i*(m-s))]
        print(f"  - {pop - df_outlier.shape[0]} ({100*(1-df_outlier.shape[0]/pop):.2f}%) samples within {i} standard deviation ")
    print("=" * 69)
    print(f"{df_outlier.shape[0]} outliers identified beyond {deviations} standard deviations")
    if verbose:
        print(df_outlier.head().T)
    if remove:
        df = df.drop(df_outlier.index)
        print(f"Removed outliers from original DataFrame. Remaining samples: {df[df.target_mask.eq(True)].shape[0]}")
    return df, df_outlier

def tag_outliers_by_neighbors(df:pd.DataFrame, features: list, n_neighbors:int=5, drop: bool=False, name:str="")-> pd.DataFrame:
    """
    Uses skl.neighbors model to identify outliers across given features
    returns df with new feature identifying outliers
    """
    outliermodel = skl.neighbors.LocalOutlierFactor(n_neighbors=n_neighbors)
    df[f'outlier_{name}'] = outliermodel.fit_predict(df[features])
    print("=" * 69, f"\n{100 * df.query(f'outlier_{name} == -1').shape[0] / df.shape[0] :.2f}% of samples identified as outliers")
    if drop is True:
        outlier_mask = df_train[df_train[f'outlier_{name}'] == -1]
        df.drop(outlier_mask.index, inplace=True)
        df.drop(f'outlier_{name}', inplace=True, axis=1)
        print(f"Removed {outlier_mask.shape[0]} outliers from original DataFrame.")
    return df

def get_cycles_from_datetime(df:pd.DataFrame, feature: str, drop:bool=False, verbose:bool=True, debug:bool=False)->pd.DataFrame:
    """
    decomposes a datetime feature into numeric and categorical features suitable for training
    ------
    requires: pandas, numpy, seaborne
    """
    def _cycle(df, feature, points):
        df[f'{feature}_{points}_sin'] = np.sin(2 * np.pi * df[feature]/points)
        df[f'{feature}_{points}_cos'] = np.cos(2 * np.pi * df[feature]/points)
        return df

    def _plot_circle(df, cyclic_features):
        plt.scatter(df[cyclic_features[0]], df[cyclic_features[1]])
        plt.xlabel(None)
        plt.xticks(())
        plt.ylabel(None)
        plt.yticks(())
        plt.show()
        
    MY_PALETTE = get_colors()
    
    if verbose:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex = True, figsize=(8,3))
        sns.histplot(data=df[df.target_mask.eq(True)], x=feature, color = MY_PALETTE[0], 
                     ax=axs[0])
        sns.histplot(data=df[df.target_mask.eq(False)], x=feature, color = MY_PALETTE[2], 
                     ax=axs[1])
        plt.show()
        
    df[f'{feature}_dummy'] = round((df[feature] - df[feature].min()).dt.days + 1, 0)
    df[f'{feature}_year'] = df[feature].dt.year.astype('int32').astype('category')
    df[f'{feature}_doy'] = df[feature].apply(lambda d: d.timetuple().tm_yday)
    df = _cycle(df, f'{feature}_doy', 366)
    df[f'{feature}_month'] = df[feature].dt.month.astype('int8')
    df= _cycle(df, f'{feature}_month', 12)
    df[f'{feature}_month'] = df[f'{feature}_month'].astype('category')
    df[f'{feature}_dom'] = df[feature].dt.day.astype('int8')
    df= _cycle(df, f'{feature}_dom', 31)
    df[f'{feature}_dom'] = df[f'{feature}_dom'].astype('category')
    df[f'{feature}_dow'] = 1 + df[feature].dt.dayofweek.astype('int8')
    df= _cycle(df, f'{feature}_dow', 7)
    df[f'{feature}_dow'] = df[f'{feature}_dow'].astype('category')

    if debug==True:
        _plot_circle(df, [f'{feature}_doy_366_sin', f'{feature}_doy_366_cos'])
        _plot_circle(df, [f'{feature}_dow_7_sin', f'{feature}_dow_7_cos'])
    
    if drop:
        df.drop(feature, inplace = True, axis = 1)
    if verbose:
        print(f"{feature} features: {[f for f in df.columns if feature in f]}")
    return df

###EDA functions

def plot_target_eda(df: pd.DataFrame, target: str, title: str='target distribution', hist: int=20) -> None:
    """
    plots simple target distribution plot
    if target is continuous (float or int with many unique values), plots histogram with KDE
    if target is categorical (object, category, bool, or int with few unique values), plots countplot
    -----------
    requires: seaborn, matplotlib, pandas
    """
    try:
        df_plot = df[df.target_mask.eq(True)][target].to_frame()
    except:
        df_plot = df[target].to_frame()
    if pd.api.types.is_float_dtype(df[target]) or (df[target].dtype == int and df[target].nunique() > hist):
        sns.histplot(df_plot[target], 
                     bins = min(df_plot[target].nunique(), 42),  # limit number of bins for large unique value counts
                     kde = True)
    else:
        sns.countplot(data=df_plot, x=target)
    plt.title(title)
    plt.yticks([])
    plt.show()

def plot_features_eda(df_: pd.DataFrame, features: list, target: str, label: str=None, 
                      sample: int=1000, y_min: float=None, y_max: float=None,
                      high_label = "", low_label = "") -> None:
    """ 
    supports feature EDA with numeric targets
    -----------
    for each numeric feature, plots:
        - distribution histogram
        - scatterplot with trendline showing relationship to target
        - boxplot showing outliers and limits by label (if label provided)
    for each categorical feature, plots:
        - distribution countplot
        - psedo-scatterplot with trendline showing relationship to target
        - donut showing variation in target by category (if label provided)
    sample limits the number of points plotted in relationship plots
    high_label and low_label are used for boxplot and donut labels when label is provided
    y_min and y_max can be set to limit the y-axis of relationship plots
    --
    for performance, limits the number of features plotted to 20
    -----------
    requires: seaborn, matplotlib, pandas, numpy
    """
    MY_PALETTE = get_colors()
    SEED = 67
    df = df_[df_.target_mask.eq(True)]
    ### Histogram for distribution of numeric feature (num plot 0)
    def _plot_num_distribution(ax, feature):
        bins = min(50, df[feature].nunique())
        sns.histplot(df[feature], ax=ax, bins = bins, discrete=True)
        ax.set_title(f'{feature} distribution')
        ax.set_yticks([])
        ax.set_ylabel("Count")
        ax.set_xlabel("")
    
    ### Countplot for distribution of categorical feature (cat plot 0)
    def _plot_cat_distribution(ax, feature, order, color_map):
        sns.countplot(data=df, x=feature, order=order, ax=ax,
                      palette=[color_map[val] for val in order])
        ax.set_title(f'{feature} distribution')
        ax.set_yticks([])
        ax.set_ylabel("Count")
        if len(order) > 8: 
            x = ax.get_xticks()
            ax.set_xticks(x, order, rotation=90)
        if len(order) > 20:
            x = ax.get_xticks()
            labels = [s if i % 5 == 0 else "" for i, s in enumerate(order)]
            ax.set_xticks(x, labels, rotation=90)
        ax.set_xlabel("")
    
    ### TODO: validate distribution plot for datetime feature (dt plot 0)
    def _plot_dt_distribution(ax, feature, target):
        df_w = df.groupby([pd.Grouper(key=f"{feature}", freq="W")])[target].count().rename(f"count_w").reset_index()
        sns.lineplot(data=df_w, x=feature, y=f"count_w", ax=ax)
        df_m = df.groupby([pd.Grouper(key=f"{feature}", freq="MS")])[target].count().rename(f"count_m").reset_index()
        sns.lineplot(data=df_m, x=feature, y=f"count_m", ax=ax)
        ax.set_title(f'{feature} distribution')
        ax.set_yticks([])
        ax.set_ylabel("Count")
        ax.set_xlabel("")

    ### scatterplot with trendline for numerical feature relationship to numeric target (num plot 1N)
    def _plot_num_relationship(ax, feature,  y_min=0, y_max=100):
        df_sampled = df.sample(n=min(sample, df.shape[0]), random_state=SEED)
        sns.regplot(data=df_sampled, x=feature, y=target, ax=ax,
                    scatter_kws={'alpha': 0.5, 's': 12}, line_kws={'color': 'xkcd:rust', 'linestyle': ":", 'linewidth': 1})
        sns.lineplot(data=df_sampled, x=feature, y=target, ax=ax, 
                     color='xkcd:rust', linewidth=1)
        ax.set_title(f'{target} vs {feature}')
        ax.set_ylabel("")
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("")

    ### density plot for numeric feature relationship to categorical target (num plot 1C)
    def _plot_num_tgt_cat_relationship(ax, feature, y_order):
        sns.histplot(data=df, stat='percent', x=feature, y=target,
                     discrete=[True, True], legend=False, ax=ax,
                     pthresh=0.002, pmax=.998, zorder=0)
        quadmesh = ax.collections[0]
        densities = quadmesh.get_array().data
        densities_flat = densities.ravel()
        x_edges = quadmesh._coordinates[0, :, 0]
        y_edges = quadmesh._coordinates[:, 0, 1]
        ny = len(y_edges) - 1
        nx = len(x_edges) - 1
        sorted_idx = np.argsort(densities_flat)
        N = min(len(x_edges)//2, 10)
        top_idx = sorted_idx[-N:]
        bottom_idx = sorted_idx[:N]
        def _annotate_bins(indices, symbol, color):
            for idx in indices:
                iy, ix = np.unravel_index(idx, (ny, nx))
                x_center = (x_edges[ix] + x_edges[ix+1]) / 2
                y_center = (y_edges[iy] + y_edges[iy+1]) / 2
                ax.text(float(x_center), float(y_center), symbol,
                        ha="center", va="center",
                        color=color, fontsize=10, zorder=2)

        _annotate_bins(top_idx, "●", 'xkcd:rust') #⊤
        _annotate_bins(bottom_idx, "⊥", 'xkcd:rust') #⊥
        ax.set_title(f'{target} vs {feature}')
        ax.set_ylabel("")
        ax.set_xlabel("")
        y = ax.get_yticks()
        labels = [""] * len(y)
        labels[0] = y_order[0]
        labels[-1] = y_order[-1]
        ax.set_yticks(y, labels, rotation=90)


    ### psedo-scatterplot with trendline for categorical feature relationship to numeric target (cat plot 1N)
    def _plot_cat_relationship(ax, feature, order, color_map, y_min=0, y_max=100):
        grouped = df.groupby(feature)
        sampled_dfs = []
        for name, group in grouped:
            if len(group) == 0:
                continue  # Skip empty groups
            frac = min(1.0, sample / len(df))
            n_samples = max(1, int(frac * len(group)))
            if len(group) == 0 or n_samples == 0:
                continue
            sampled_dfs.append(group.sample(n=n_samples, random_state=SEED))
        if sampled_dfs:
            df_sampled = pd.concat(sampled_dfs)
            sns.stripplot(data=df_sampled, x=feature, y=target, order=order, ax=ax, zorder = 1, 
                              palette=[color_map[val] for val in order], alpha=0.5, jitter=True)
        sns.pointplot(data=df, x=feature, y=target, order=order, ax=ax, zorder = 2, 
                      color='xkcd:rust', errorbar = None)

        if len(df[target].unique()) > 5:
            for i, val in enumerate(order):
                subset = df[df[feature] == val][target].dropna()
                q25, q75 = subset.quantile([0.25, 0.75])
                ax.vlines(x=i, ymin=q25, ymax=q75, color='xkcd:rust', linewidth=2,  zorder = 3)
        
        ax.set_title(f'{target} vs {feature}')
        if len(order) > 8: 
            x = ax.get_xticks()
            ax.set_xticks(x, order, rotation=90)
        if len(order) > 20:
            x = ax.get_xticks()
            labels = [s if i % 5 == 0 else "" for i, s in enumerate(order)]
            ax.set_xticks(x, labels, rotation=90)
        ax.set_ylabel("")
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("")

    ### density plot for categorical feature relationship to categorical target (cat plot 1C)
    def _plot_cat_tgt_cat_relationship(ax, feature, order, y_order):
        sns.histplot(data=df, stat='percent', x=feature, y=target, zorder=0, 
                                  legend=False, discrete=[True,True], pthresh=0.02, pmax=.98, ax=ax)

        mode_points = df.groupby(feature)[target].agg(lambda x: x.value_counts().idxmax()).reset_index()
        sns.pointplot(data=mode_points, x=feature, y=target, zorder=1,
                                    color='xkcd:rust', errorbar=None)
        ax.set_title(f'{target} vs {feature}')
        ax.set_ylabel("")
        ax.set_xlabel("")
        if len(order) > 8: 
            x = ax.get_xticks()
            ax.set_xticks(x, order, rotation=90)
        if len(order) > 20:
            x = ax.get_xticks()
            labels = [s if i % 5 == 0 else "" for i, s in enumerate(order)]
            ax.set_xticks(x, labels, rotation=90)
        y = ax.get_yticks() 
        labels = [""] * len(y)
        labels[0] = y_order[0]
        labels[-1] = y_order[-1]
        ax.set_yticks(y, labels, rotation=90)

        ### TODO: validate datetime feature relationship to target (dt plot 1N)
    def _plot_dt_relationship(ax, feature, y_min=0, y_max=100):
        df_sampled = df.sample(n=min(sample, df.shape[0]), random_state=SEED)
        sns.scatterplot(data=df_sampled, x=feature, y=target, ax=ax, zorder=0, alpha=0.5)
        df_w = df.groupby([pd.Grouper(key=f"{feature}", freq="W")])[target].mean().rename(f"mean_w").reset_index()
        sns.lineplot(data=df_w, x=feature, y=f"mean_w", ax=ax)
        df_m = df.groupby([pd.Grouper(key=f"{feature}", freq="MS")])[target].mean().rename(f"mean_m").reset_index()
        sns.lineplot(data=df_m, x=feature, y=f"mean_m", ax=ax)
        ax.set_title(f'{target} mean vs {feature}')
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_ylim(y_min, y_max)

    ### TODO: add datetime feature relationship to categorical target (dt plot 1C)
    def _plot_dt_tgt_cat_relationship(ax, feature, y_order):
        #TODO validate this - is this the right plot?
        for t in y_order:
            df_w = df.groupby([pd.Grouper(key=f"{feature}", freq="W")])[target==t].count().rename(f"count_{t}_w").reset_index()
            sns.lineplot(data=df_w, x=feature, y=f"count_{t}_w", ax=ax)
        ax.set_title(f'Count {target} values vs {feature}')
        ax.set_ylabel("")
        ax.set_xlabel("")
        

    ### boxplot shows outliers and limits by label  (num plot 2)
    def _plot_num_boxplot(ax, feature, label = None, top_label="", bottom_label=""):
        if label == None:
            sns.boxplot(x = df[feature], ax=ax)
            ax.set_title(f'{feature} outliers')
        else:
            cats = sorted(df[label].dropna().unique().tolist(), reverse=True)
            sns.boxplot(x = df[feature], palette=MY_PALETTE , ax=ax, legend = False, gap = .1,
                        hue = df[label], hue_order = cats)
            ax.set_title(f'{feature} by target cut')
            ax.set_xlabel("")
            if top_label == "" and bottom_label =="":
                top_label, bottom_label = cats[0], cats[-1]
            ax.text(df[feature].min(), -0.45, top_label, ha='left', va='center', fontsize=8, color = 'black')
            ax.text(df[feature].min(), 0.45, bottom_label, ha='left', va='center', fontsize=8, color = 'black')
        ax.set_yticks([])

    ### donut shows variation in target by category  (cat plot 2)
    def _plot_cat_donut(ax, feature, label, order, color_map, inner_label="", outer_label=""):
        if label == None: return
        cats = sorted(df[label].dropna().unique().tolist(), reverse=True)
        ring_width = 0.7 / len(cats)
        if inner_label == "" and outer_label =="":
            outer_label, inner_label  = cats[0], cats[-1]
        for i, cat in enumerate(cats):
            value_counts = df[df[label] == cat][feature].value_counts()
            sorted_counts = value_counts.reindex(order).dropna()
            if len(order) > 20:
                labels = [s if i % 5 == 0 else "" for i, s in enumerate(sorted_counts.index)]
            else: labels = sorted_counts.index
            slice_colors = [color_map[val] for val in sorted_counts.index]
            radius = 1 - ring_width * i
            ax.pie(sorted_counts, radius=radius, colors=slice_colors,
                   wedgeprops=dict(width=ring_width, edgecolor='w'),
                   labels=labels if i == 0 else None)
            ax.set_title(f'{feature} by target cut')
            ax.text(0, 0, inner_label, ha='center', va='center', fontsize=8, color = 'xkcd:steel grey')
            ax.text(-1.3, -1.3, outer_label, ha='left', va='center', fontsize=8, color = 'xkcd:steel grey')

    ### TODO: what should this plot look like for datetime features (dt plot 2)
    def _plot_dt_lagplot(ax, feature, lag, label):
        #TODO what does this look like? maybe a lag plot?
        pass

    ### limit number of features plotted/size of plot
    f = min(20, len(features))
    features = features[:f]
    if f > 20: print("Plotting first 20 features")
    
    ### gridspec to build plot layout
    fig = plt.figure(figsize=(10, f * 3))
    gs = mpl.gridspec.GridSpec(f, 3, figure=fig, hspace=0.4)
    row_anchors = []

    ### determine nature of target
    tgt_cat = (df[target].dtype == "O" or df[target].dtype == bool or 
               df[target].dtype == "category" or df[target].nunique() < 4)
    if tgt_cat:
        df[target] = df[target].astype(str).astype('category')
        y_order = sorted(df[target].unique().tolist(), reverse=True)
        df[target] = pd.Categorical(df[target], categories=y_order, ordered=True)
    else:
        if not y_min: y_min = df[target].min()
        if not y_max: y_max = df[target].max()

    for i, feature in enumerate(features):
        ax0 = fig.add_subplot(gs[i, 0])
        row_anchors.append(ax0)
        ### for each feature determine applicable plot selection
        if df[feature].dtype == 'O':
            try:
                df[feature] = pd.to_datetime(df[feature])
            except Exception:
                pass
        is_cat = (df[feature].dtype == "O" or 
                  df[feature].dtype == bool or 
                  df[feature].dtype == "category" or 
                  (df[feature].dtype=='int' and df[feature].nunique() < 4))
        is_dt = True if pd.api.types.is_datetime64_any_dtype(df[feature]) else False
        if is_dt:
            #distribution plot (0)
            _plot_dt_distribution(ax0, feature, target)
            #target relationship plot(1)
            if tgt_cat: _plot_dt_tgt_cat_relationship(fig.add_subplot(gs[i, 1]), feature, y_order)
            else: _plot_dt_relationship(fig.add_subplot(gs[i, 1]), feature, y_min=y_min, y_max=y_max)
            #target distribution by feature plot (2)
            if label != None:
                _plot_dt_lagplot(fig.add_subplot(gs[i, 2]), feature, lag, label)
        elif is_cat:
            order = sorted(df[feature].dropna().unique().tolist())
            df[feature] = pd.Categorical(df[feature], categories=order, ordered=True)
            color_map = get_colors(color_keys=order, get_cmap=True, n_hues=6, n_sats=5)
            #distribution plot (0)
            _plot_cat_distribution(ax0, feature, order, color_map)
            #target relationship plot(1)
            if tgt_cat: _plot_cat_tgt_cat_relationship(fig.add_subplot(gs[i, 1]), feature, order, y_order)
            else: _plot_cat_relationship(fig.add_subplot(gs[i, 1]), feature, order, color_map, y_min=y_min, y_max=y_max)
            #target distribution by feature plot (2)
            if label != None:
                _plot_cat_donut(fig.add_subplot(gs[i, 2]), feature, label, order, color_map,
                                inner_label=low_label, outer_label=high_label)
        else:
            #distribution plot (0)
            _plot_num_distribution(ax0, feature)
            #target relationship plot(1)
            if tgt_cat: _plot_num_tgt_cat_relationship(fig.add_subplot(gs[i, 1]), feature, y_order)
            else: _plot_num_relationship(fig.add_subplot(gs[i, 1]), feature, y_min=y_min, y_max=y_max)
            if label != None:
                _plot_num_boxplot(fig.add_subplot(gs[i, 2]), feature, label, 
                                                  top_label=high_label, bottom_label=low_label)
    ### add tear lines between features
    for i in range(f - 1):
        bottom_y = row_anchors[i].get_position().y0
        top_y = row_anchors[i + 1].get_position().y1
        y_pos = (bottom_y + top_y) / 2
        line = mpl.lines.Line2D([0.05, 0.95], [y_pos, y_pos], transform=fig.transFigure,
                      color='black', linewidth=0.5, linestyle='--')
        fig.add_artist(line)

    plt.show()

def plot_pairplot(df: pd.DataFrame, features: list, sample: int=250, title: str="", **kwargs) -> None:
    """
    pairplot for feature to feature comparisons
    -----------
    plots:
        - pairwise scatterplots for numeric features
        - kde histograms on the diagonal
        - contour lines on the lower triangle to show density of points in scatterplots
    for performance, limits scatterplots to a sample of the data
    -----------
    requires: seaborn, matplotlib, pandas
    """
    print("=" * 69)
    plot_df = df[features].sample(n = min(sample, df.shape[0]), random_state=69)
    g = sns.pairplot(plot_df, diag_kind="kde", **kwargs)
    g.map_lower(sns.kdeplot, levels=4, color="xkcd:slate")
    g.figure.suptitle(title, x = 0.98, ha = 'right', y=1.01)
    plt.show()

def print_pca_loadings(df: pd.DataFrame, features: list, filter_small: bool=True) -> None:
    """
    prints PCA loadings for selected features in df
    -----------
    useful for understanding relationships in the data and informing feature engineering decisions
    -----------
    requires: pandas, scikit learn
    """ 
    X = df[df.target_mask.eq(True)][features[:10]]
    X = (X - X.mean()) / X.std()  # standardize features before PCA
    pca = skl.decomposition.PCA()
    X_pca = pca.fit_transform(X)
    component_names = [f"PCA{i+1}" for i in range(X_pca.shape[1])]
    loadings = pd.DataFrame(
        pca.components_.T,         # transpose the matrix of loadings
        columns=component_names,   # so the columns are the principal components
        index=X.columns,           # and the rows are the original features
    )
    if filter_small:
        loadings[(loadings > -0.1) & (loadings < 0.1)] = ""
    print(loadings)

def calculate_score(actual, predicted, metric='rmse')-> float:
    """ 
    calculates score based on metric or task
    simplifies calling scikit learn metrics by allowing flexible metric names
    -----------
    returns: metric score
    for rmse use 'rmse' or 'regression'
    for accuracy use 'accuracy' or 'classification'
    for roc_auc use 'roc_auc' or 'probability_roc_auc'
    -----------
    requires: scikit learn
    """
    if metric in ['rmse', 'regression', 'regression_rmse']:
        return skl.metrics.root_mean_squared_error(actual, predicted)
    elif metric in ['mae', 'regression_mae']:
        return skl.metrics.mean_absolute_error(actual, predicted)
    elif metric in ['r2', 'regression_r2']:
        return skl.metrics.r2_score(actual, predicted)
    elif metric in ['mape', 'regression_mape']:
        return skl.metrics.mean_absolute_percentage_error(actual, predicted)
    elif metric in ['rmse_log', 'rmsle', 'regression_log_rmse', 'regression_rmsle']:
        predicted = np.clip(predicted, 0, None)
        return skl.metrics.root_mean_squared_log_error(actual, predicted)
    elif metric in ['accuracy', 'classification', 'classification_accuracy']:
        return skl.metrics.accuracy_score(actual, predicted)
    elif metric in ['f1', 'classification_f1']:
        return skl.metrics.f1_score(actual, predicted)
    elif metric in ['precision', 'classification_precision']:
        return skl.metrics.precision_score(actual, predicted)
    elif metric in ['roc_auc', 'probability', 'probability_roc_auc', 'classification_roc_auc']:
        return skl.metrics.roc_auc_score(actual, predicted)
    elif metric in ['log_loss', 'probability_log_loss', 'classification_log_loss']:
        return skl.metrics.log_loss(actual, predicted)
    elif metric in ['brier', 'probability_brier', 'brier_score']:
        return skl.metrics.brier_score_loss(actual, predicted)
    else:
        raise ValueError("""***UNSUPPORTED METRIC***\n
                         Supported regression metrics: 'rmse', 'mae', 'r2', 'mape', 'rmse_log' \n
                         Supported classification metrics: 'accuracy', 'f1', 'precision', 'roc_auc', 'log_loss', 'brier'""")

def plot_training_results(X_t, X_v, y_t, y_v, y_p, task: str='regression')-> None:
    """
    plots training results with reference model for comparison
    -----------
    fits a base model to the training data (X_t, y_t) to provide a reference point for evaluating the trained model predictions
    scores the base model predictions and trained model predictions (y_p) against validation data (y_v) 
    based on the task (regression, classification, or probability)
    plots:
    for regression: actual vs predicted scatterplots for trained model and ridge regression, distribution of predictions vs actuals, and residual distribution
    for classification: confusion matrices for trained model and gaussian naive bayes, distribution of predicted probabilities
    for probability: ROC curve comparing trained model and gaussian naive bayes, distribution of predicted probabilities, and confusion matrix for predicted probabilities
    -----------
    requires: pandas, scikit learn, matplotlib, numpy
    """
    if task.startswith("regression"): base_model = skl.linear_model.Ridge()
    else: base_model = skl.naive_bayes.GaussianNB()
    numeric_features = [f for f in X_t.columns.tolist() if 
                        X_t[f].dtype != "object" and 
                        X_t[f].dtype != "string" and
                        X_t[f].dtype != "category"]
    
    base_model.fit(X_t[numeric_features], y_t)
    
    #TODO validate reshape with multiclass proba
    if task.startswith("probability"):
        y_base = base_model.predict_proba(X_v[numeric_features])[:, 1].reshape(-1, 1)
    else:
        y_base = base_model.predict(X_v[numeric_features]).reshape(-1, 1)
    
    def plot_regression_resid(ax):
        skl.metrics.PredictionErrorDisplay.from_predictions(y_v[:1000], y_base[:1000], kind = 'actual_vs_predicted',
                                                            scatter_kwargs={"color":'xkcd:gold', "alpha":0.8},
                                                            ax = ax)
        skl.metrics.PredictionErrorDisplay.from_predictions(y_v[:1000], y_p[:1000], kind = 'actual_vs_predicted', 
                                                            scatter_kwargs={"alpha":0.8},
                                                            line_kwargs={"color":'xkcd:dusty rose'},
                                                            ax = ax)
        metric = 'rmse' if task=="regression" else task.split("_")[1]
        ax.set_title(f"Trained Model {calculate_score(y_v, y_p, metric = metric):.4f} vs Ridge {calculate_score(y_v, y_base, metric = metric):.4f} {metric.upper()}")

    def plot_classification_cm(ax, predictions=y_p, title = "Trained", show_values=True):
        cmap='bone_r' if len(np.unique(predictions)) < 4 else 'cividis'
        skl.metrics.ConfusionMatrixDisplay.from_predictions(y_v, predictions, cmap=cmap, include_values=show_values,
                                                            normalize='all', colorbar=False, ax=ax)
        ax.invert_yaxis()
        ax.set_title(f"{title} Model Accuracy {100*calculate_score(y_v, predictions, metric = 'accuracy'):.1f}%")

    def plot_classification_roc(ax):
        skl.metrics.RocCurveDisplay.from_predictions(y_v, y_p, ax=ax, name="Trained Model")
        skl.metrics.RocCurveDisplay.from_predictions(y_v, y_base, name="GaussianNB", ax=ax)
        ax.set_title("ROC Curve")

    def plot_distribution(ax, hist=True):
        if hist == True:
            ax.hist(y_v, bins=min(50,len(np.unique(y_v))), color='xkcd:silver', alpha=0.8, density = True)
            ax.hist(y_p, bins=min(50,len(np.unique(y_v))), color='xkcd:ocean blue', alpha=0.9, density = True)
        else: 
            sns.countplot(data=y_v, color='xkcd:silver', alpha=0.8, ax=ax)
            sns.countplot(data=y_p, color='xkcd:ocean blue', alpha=0.9, ax=ax)
        ax.set_ylabel("Probability Density")
        ax.set_title("Prediction vs Training Distribution")
        ax.set_yticks([])

    def plot_residuals(ax):
        residuals = y_p - y_v
        ax.hist(residuals, bins=min(50,2+len(np.unique(residuals))), color='xkcd:dull green', alpha=0.9)
        ax.set_title("Residual Distribution")
        ax.set_yticks([])
        ax.set_ylabel("Count")

    fig = plt.figure(figsize=(9, 6))
    gs = mpl.gridspec.GridSpec(2, 3, figure=fig)

    if task.startswith("regression"):
        plot_regression_resid(fig.add_subplot(gs[:, :2]))
        plot_distribution(fig.add_subplot(gs[0, 2]))
        plot_residuals(fig.add_subplot(gs[1, 2]))
            
    elif task.startswith("classification"):
        plot_classification_cm(fig.add_subplot(gs[:, :2]))
        plot_distribution(fig.add_subplot(gs[0, 2]), hist=False)
        plot_classification_cm(fig.add_subplot(gs[1, 2]), 
                               predictions = y_base, 
                               show_values=True if len(np.unique(y_v)) < 4 else False, 
                               title = "GaussianNB")
    
    elif task.startswith("probability"):
        plot_classification_roc(fig.add_subplot(gs[:, :2]))
        plot_distribution(fig.add_subplot(gs[0, 2]))
        plot_classification_cm(fig.add_subplot(gs[1, 2]), predictions=np.round(y_p))

    plt.tight_layout()
    plt.show()

def check_categoricals(df: pd.DataFrame, features: list, pct_diff: float=0.1)-> None:
    """
    checks that categorical features have consistent unique values and 
    similar distributions between training and testing data
    -----------
    for each feature, checks that unique values in training and testing data are the same
    prints any unique values that appear in one set but not the other as potential noise
    checks that the percentage of each unique value in training and testing is within pct_diff threshold 
    prints any that exceed the threshold as potential consistency issues
    -----------
    requires: pandas
    """
    df_train = df[df.target_mask.eq(True)]
    df_test = df[df.target_mask.eq(False)]
    
    def _print_consistency(feature, feature_values, df_train=df_train, df_test=df_test, pct_diff=pct_diff):
        dict_index = {}
        for v in feature_values:
            try:
                train_pct = 100*df_train.groupby(feature)[feature].count()[v] / df_train.shape[0]
            except:
                train_pct = 0
            try:
                test_pct = 100*df_test.groupby(feature)[feature].count()[v] / df_test.shape[0]
            except:
                test_pct = 0
            delta = abs(train_pct - test_pct)
            dict_index[v] = {'Train': train_pct, 'Test': test_pct, 'Difference': delta}
        df_plot = pd.DataFrame.from_dict(dict_index, orient="index") 
        if df_plot[df_plot.Difference.ge(pct_diff)].shape[0] == 0:
            print(f"✔️ {pct_diff}% consistent check passes for {feature}")
        else:
            print(f"❌ {pct_diff}% consistent check FAILED for {feature}")
            print(df_plot[df_plot.Difference.ge(pct_diff)])
        print(f"    - {df_train[feature].nunique()} unique values in {feature} training data")
   
    for feature in features:
        train_v = df_train[feature].unique()
        test_v = df_test[feature].unique()
        train_noise = [f for f in train_v if f not in test_v]
        test_noise = [f for f in test_v if f not in train_v]
        if train_noise == [] and test_noise == []:
            if len(train_v) >= 2:
                _print_consistency(feature, train_v)
            else:
                print(f"{feature} is trivial, recommend dropping {feature}")
        else:
            print(f"❌ unique feature check FAILED for {feature}")
            if test_noise != []:
                print(f"*** Warning {feature} has test values that do not appear in training data! ***")
                print(f"{feature} values: {test_noise} appear in testing, but not training data")
            if train_noise != []:
                print(f"{feature} values: {train_noise} appear in training, but not test data")
            all_v = list(set(train_v) | set(test_v))
            _print_consistency(feature, all_v)

def plot_feature_corr(df, features, target=None):
    plot_features = [f for f in features if
                     (df[f].dtype == 'int' or df[f].dtype == 'float')]
    if target is not None:
        plot_features.insert(0, target)
    plt.figure(figsize=(8,6))
    sns.heatmap(data=df[plot_features].corr(), 
        mask=np.tril(df[plot_features].corr()), 
        annot=True if (len(plot_features)<12) else False, 
        fmt='.2f', 
        square=True,
        cbar=False,
        cmap='cividis', #coolwarm is a good cmap
        vmin = -1, vmax = 1,
        linewidth=1, linecolor='white')

    plt.xticks(())
    plt.xlabel("")
    plt.ylabel("")
    plt.show()

def check_all_features_scaled(df: pd.DataFrame, targets:list)-> None:
    features = [f for f in df.columns if f not in targets and 
                df[f].dtype != 'category' and df[f].dtype!='bool']
    features_alt = [f for f in df.columns if f not in targets and 
                (df[f].dtype == 'float' or df[f].dtype == 'int')]
    if features == features_alt:
        unscaled_features = [f for f in features if (df[f].mean() > 1 or df[f].mean() <-1)]
        if unscaled_features == []:
            print("All features scaled")
        else:
            print(f"Consider scaling: {unscaled_features}")
    elif features==[]:
        print("No numeric features in DataFrame")
    else:
        print(f"Object features: {[f for f in features if f not in features_alt]}")

### Train and evaluate
def train_and_score_model(X_train: pd.DataFrame, X_val:pd.DataFrame, 
                          y_train: pd.Series, y_val:pd.Series,
                          model, task: str="regression", 
                          verbose: bool=True, 
                          TargetTransformer=None):
    """
    trains a model and returns trained model & score
    -----------
    model: a scikit learn compatible model with fit and predict methods
    task: "regression", "classification", or "probability" to determine prediction and scoring method
    returns:
    - trained model
    - score based on task
    -----------
    requires: pandas, scikit learn, numpy, matplotlib
    """
    model.fit(X_train, y_train)
    if task.startswith("regression") or task.startswith("classification"): y_predict = model.predict(X_val)
    elif task.startswith("probability"): y_predict = model.predict_proba(X_val)[:, 1]
    else: 
        print(f"Unknown task {task}")
        return model, None
    if TargetTransformer != None:
        y_t = TargetTransformer.inverse_transform(y_train.values.reshape(-1, 1))
        y_v = TargetTransformer.inverse_transform(y_val.values.reshape(-1, 1))
        y_p = TargetTransformer.inverse_transform(y_predict.reshape(-1, 1))
    else:
        y_t = np.array(y_train).reshape(-1, 1)
        y_v = np.array(y_val).reshape(-1, 1)
        y_p = np.array(y_predict).reshape(-1, 1)
    score = calculate_score(y_v, y_p, metric = task)
    print(f"***  model score:  {score:.4f}  ***")
    if verbose == True: 
        plot_training_results(X_train, X_val, y_t, y_v, y_p,
                              task=task) 
    return model, score

def get_feature_importance(X_train: pd.DataFrame, X_val: pd.DataFrame,
                           y_train: pd.DataFrame, y_val: pd.DataFrame, 
                           verbose: bool=True, task: str="regression"):
    """
    gets feature importance by training an LightGBM model
    -----------
    returns: a Series of feature importances sorted in descending order
    trains a LightGBM model on the training data and evaluates on the validation data
    -----------
    requires: pandas, lightgbm, scikit learn
    """
    if task.startswith("regression"): model = lgb.LGBMRegressor(verbose=-1)
    else: model = lgb.LGBMClassifier(verbose=-1)
    ModelFeatureImportance, _ = train_and_score_model(X_train, X_val, y_train, y_val,
                                                      model=model,
                                                      verbose=False, task=task)
    ds = pd.Series(ModelFeatureImportance.feature_importances_, name="importance", index=X_train.columns)
    ds.sort_values(ascending=False, inplace = True)
    print("=" * 69)
    print(f"  ***  Top feature is: {ds.index[0]}  *** \n")
    ds[:10].plot(kind = 'barh', title = f"Top {min(10, len(ds))} of {len(ds)} Features")
    if verbose:
        if len(ds) > 50: features_to_show = 10
        elif len(ds) > 20: features_to_show = 5
        else: features_to_show = 3
        print("=" * 69)
        print(f"  Top Features:")
        print(ds.head(features_to_show))
        print("=" * 69)
        print(f"  Bottom Features:")
        print("=" * 69)
        print(ds.tail(features_to_show))
        print("=" * 69)
        print(f"Zero importance features: {(ds == 0).sum()} of {len(ds.index)}")
    return ds

def get_feature_mutual_info(X: pd.DataFrame, y: pd.DataFrame, task:str='regression', verbose: bool=True):
    """
    gets feature mutual importance
    -----------
    returns: a Series of mutual information sorted in descending order
    -----------
    requires: pandas, scikit learn
    """
    if X.shape[0] > 10000:
        sample_size = 10000
        X = X.sample(n=sample_size, random_state=69)
        y = y.loc[X.index]
    if task.startswith("regress"):
        mi_scores = skl.feature_selection.mutual_info_regression(X, y)
    else:
        mi_scores = skl.feature_selection.mutual_info_classif(X, y)
    ds = pd.Series(mi_scores, name="information", index=X.columns)
    ds.sort_values(ascending=False, inplace=True)
    print("=" * 69)
    print(f"  ***  Top feature is: {ds.index[0]}  *** \n")
    ds[:10].plot(kind = 'barh', title = f"Top {min(10, len(ds))} of {len(ds)} Features")
    if verbose:
        if len(ds) > 50: features_to_show = 10
        elif len(ds) > 20: features_to_show = 5
        else: features_to_show = 3
        print("=" * 69)
        print(f"  Top {task} Features:")
        print(ds.head(features_to_show))
        print("=" * 69)
        print(f"  Bottom {task} Features:")
        print("=" * 69)
        print(ds.tail(features_to_show))
        print("=" * 69)
        print(f"Zero info features: {(ds == 0).sum()} of {len(ds.index)}")
    return ds

def study_model_hyperparameters(df: pd.DataFrame, features: list, target: str, study_model: str, 
                                     metric: str='classification', direction: str='maximize',
                                     n_trials: int=20, timeout: int=1200, sample_size: int=25000,
                                     CORES: int=4, DEVICE: str='cpu', verbose: bool=True)-> dict:
    '''
    studies impact of hyperparameters on study models
    --------
    returns a dictionary of "good" hyperparamters for a given study_model, features, target
    supports study_models for: 
        'hgb' : skl.ensemble.HistGradientBoostingClassifier,
        'rf' : skl.ensemble.RandomForestClassifier,
        'lr' : skl.linear_model.LogisticRegression,
        'lgb' : lgb.LGBMClassifier,
        'xgb' : xgb.XGBClassifier, 
        'catb' : catb.CatBoostClassifier
    --------
    requires: numpy, pandas, optuna, plotly, scikit learn, lightgbm, xgboost, catboost
    '''
    SEED=69
    cat_list = [f for f in features if df[f].dtype == "category"]

    def _study_objective(trial, study_model, X_train, y_train, X_val, y_val, 
                         cat_features=cat_list, metric=metric):
        if 'lgb' in study_model or 'light' in study_model:
            study_params = {
                'n_estimators': trial.suggest_int('n_estimators', 96, 320, step=16),      # Default=100
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0042, .42),  # Default=0.1
                'num_leaves':trial.suggest_int('num_leaves', 20, 96, step=4),             # Default=31
                'reg_alpha':  trial.suggest_loguniform('reg_alpha', 0.00069, .069),       # Default=0.0
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.00069, .069),      # Default=0.0
            }
            study_params['n_jobs']=CORES
            study_params['verbose'] = -1
            if metric.startswith("probability") or metric.startswith("classification"): 
                study_params['objective'] = 'binary'
                study_params['metric'] = 'auc'
                model = lgb.LGBMClassifier(**study_params, random_state=SEED)
            else:
                study_params['objective'] = 'regression'
                study_params['metric'] = 'rmse'
                model = lgb.LGBMRegressor(**study_params, random_state=SEED)

        elif 'xgb' in study_model:
            study_params = {
                'n_estimators': trial.suggest_int('n_estimators', 96, 640, step=16),  
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0042, .42),
                'max_leaves':trial.suggest_int('max_leaves', 20, 192, step=4),
                'max_bins':trial.suggest_int('max_bins', 16, 208, step=4),
                'reg_alpha':  trial.suggest_loguniform('reg_alpha', 0.00069, .069),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.00069, .069),
                'gamma': trial.suggest_float('gamma', 0.0042, 0.042),
            }
            study_params['device']=DEVICE
            study_params['verbosity'] = 0
            study_params['enable_categorical'] = True
            if metric.startswith("probability") or metric.startswith("classification"): 
                model = xgb.XGBClassifier(**study_params, random_state=SEED)
            else:
                model = xgb.XGBRegressor(**study_params, random_state=SEED)

        elif 'catb' in study_model:
            study_params = {
                'n_estimators': trial.suggest_int('n_estimators', 96, 320, step=16),  
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0042, 0.42),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.0001, 0.1),
                'model_size_reg': trial.suggest_loguniform('model_size_reg', 0.00042, 0.42),
                'border_count': trial.suggest_int('border_count', 24, 128, step=4)}
            study_params['task_type']='GPU' if DEVICE == "cuda" else 'CPU'
            study_params['cat_features'] = cat_features
            study_params['verbose'] = 0
            if metric.startswith("probability") or metric.startswith("classification"): 
                model = catb.CatBoostClassifier(**study_params, random_state=SEED)
            else:
                model = catb.CatBoostRegressor(**study_params, random_state=SEED)
        
        elif 'hgb' in study_model or 'hist' in study_model:
            study_params = {
                'max_iter': trial.suggest_int('max_iter', 1000, 1066),                               # Default=100
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0069, 0.69),            # Default=0.1
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 128, step=4),              # Default=31
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 16, 36),                   # Default=20
                'l2_regularization': trial.suggest_loguniform('l2_regularization', 0.000001, 0.01),  # Default=0.0
            }
            study_params['early_stopping'] = True
            if metric.startswith("probability") or metric.startswith("classification"): 
                model = skl.ensemble.HistGradientBoostingClassifier(**study_params, random_state=SEED)
            else:
                model = skl.ensemble.HistGradientBoostingRegressor(**study_params, random_state=SEED)
                
        
        elif 'rf' in study_model or 'forest' in study_model or 'random' in study_model:
            study_params = {
                'n_estimators': trial.suggest_int('n_estimators', 67, 167),                           # Default=100
                'criterion': trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"]), # Default='squared_error'
                'max_depth': trial.suggest_int('max_depth', 4, 16, step=2),                            # Default=None
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),                    # Default=2
            }
            # need work to incorporate cuml
#            if DEVICE == "cuda": model = cuml.ensemble.RandomForestClassifier(**study_params, output_type="numpy")
#            else: 
            study_params['n_jobs'] = CORES
            if metric.startswith("probability") or metric.startswith("classification"): 
                model = skl.ensemble.RandomForestClassifier(**study_params, random_state=SEED)
            else:
                model = skl.ensemble.RandomForestRegressor(**study_params, random_state=SEED)

        elif 'lr' in study_model or 'log' in study_model or 'linear' in study_model:
            if metric.startswith("probability") or metric.startswith("classification"): 
                study_params = {
                    'C': trial.suggest_loguniform('C', 0.067, 6.7),                          # Default=1 
                    'max_iter': trial.suggest_int('max_iter', 99, 291, step=24),             # Default=100
                    'solver': trial.suggest_categorical('solver', ["lbfgs", "sag", "saga"]), # Default='lbfgs'
                }
                study_params['n_jobs'] = CORES
                model = skl.linear_model.LogisticRegression(random_state=SEED)
            else:
                study_params = {
                    'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                    'normalize': trial.suggest_categorical('normalize', [True, False]),
                }
                study_params['n_jobs'] = CORES
                model = skl.linear_model.LinearRegression(**study_params)
        else:
            raise ValueError("Unrecognized model type")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val) 
        print(f"Trial {trial.number} {metric} ")
        return  calculate_score(y_val, y_pred, metric=metric)

    def _add_static_params(study_params, study_model, cat_features=cat_list):
        if 'lgb' in study_model:
            study_params['n_jobs']=CORES
            study_params['verbose'] = -1
        elif 'xgb' in study_model:
            study_params['device']=DEVICE
            study_params['verbosity'] = 0
            study_params['enable_categorical'] = True
        elif 'catb' in study_model:
            study_params['task_type']='GPU' if DEVICE == "cuda" else 'CPU'
            study_params['cat_features'] = cat_features
            study_params['verbose'] = 0            
        elif 'hgb' in study_model:
            study_params['early_stopping'] = True
        elif 'rf' in study_model or 'log' in study_model or 'linear' in study_model or 'lr' in study_model:
            study_params['n_jobs'] = CORES
        return study_params

    # Main loop    
    print("=" * 69)
    print(f"Studying {study_model} hyperparameters")
    print("=" * 69)
    X_train, y_train, X_val, y_val, _, _ = split_training_data(df, features, target, validation_size = 0.2)
    idx = X_train.sample(n=min(sample_size, X_train.shape[0]), random_state=69).index
    X_t = X_train.loc[idx]
    y_t = y_train.loc[idx]
    
    tic = time()
    study = optuna.create_study(direction=direction)
    m = metric.split("_")[0] if "_" in metric else metric
    study.optimize(lambda trial: _study_objective(trial, study_model, X_t, y_t, X_val, y_val, metric=m),
                   n_trials=n_trials, timeout=timeout)
    toc = time()
    trial = study.best_trial
    if verbose == True:
        fig = optuna.visualization.plot_optimization_history(study)
        go.io.show(fig)
        fig = optuna.visualization.plot_slice(study)
        go.io.show(fig)
        print(f"Average Trial Duration {(toc - tic)/n_trials:.3f}sec")
        print(f"Number of finished trials: {len(study.trials)}") 
        print(f"Best trial:\n  {metric} value: {trial.value}\n  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("=" * 69)
        print(f"Best {study_model} params: {study.best_params}")
    print("=" * 69)
    params = _add_static_params(study.best_params, study_model)
    return params

def get_ready_models(df: pd.DataFrame, features: list, target:str, base_models:dict, 
                              n_features: int=3, task: str='regression', direction: str='minimize',
                              n_trials: int=22, timeout: int=1200,
                              CORES: int=4, DEVICE: str='cpu', 
                              hyper_params: dict=None, verbose: bool=True):
    """
    enables training multiple models with different feature subsets and hyperparameters
    instantiates each models with hyperparameters
        use hyper_params to provide a dictionary of hyperparameters for each model
        if model hyper_params are not provided, uses optuna to find good hyperparameters for each model
    identifies different feature subsets for each model 
        use n_features = 1 to use all features for each model
        use n_features > 1 to use different 1/n subsets of features for each model
    --------
    returns:
        a dictionary of models ready for training and
        a dictionary of feature subsets used for training each model
    --------
    requires: numpy, pandas
    optional: optuna, plotly, scikit learn, lightgbm, xgboost, catboost
    """
    def _every_nth(seq, n, start=0): return seq[start::n]

    params = {}
    training_features = {}

    for i, (k, model_cls) in enumerate(base_models.items()):
        if n_features <= 1 or n_features >= len(features) or n_features is None:
            feats = features
        else:
            feats = _every_nth(features, n_features, start = i//n_features)
        training_features[k] = feats
        if hyper_params is not None and k in hyper_params:
            params[k] = hyper_params[k]
        else:
            params[k] = study_model_hyperparameters(
                df, feats, target, k, sample_size=25000,
                metric=task, direction=direction,
                n_trials=n_trials, timeout=timeout,
                CORES=CORES, DEVICE=DEVICE, verbose=verbose
            )

    models = {
        k: model_cls(**params[k])
        for k, model_cls in base_models.items()
    }
    return models, training_features

def cv_train_models(df: pd.DataFrame, features: dict, target: str, models: dict,
                    task: str = "regression", folds: int = 7, meta_model=None,
                    TargetTransformer=None, verbose: bool = True):
    """
    trains models with cross validation and returns trained models and a stacking meta model
    -----------
    for each model in models, trains with cross validation using the corresponding feature subset in features
    returns a dictionary of trained models and a meta stacking model
    prints OOF validation score for each model
    task determines the prediction and scoring method used for validation
    -
    note: when TargetTransformer is used, meta model is trained on inverse transformed target values.
          Meta model output will NOT REQUIRE inverse transform
    -----------
    requires: numpy, pandas, scikit learn
    optional: lightgbm, xgboost, catboost
    """
    def _get_all_features(features=features):
        list_of_lists = [f for f in features.values()]
        flat = []
        seen = set()
        for sub in list_of_lists:
            for item in sub:
                if item not in seen:
                    seen.add(item)
                    flat.append(item)
        return flat

    print("=" * 69)
    print(f"Training {len(models.keys())} Models")
    print("=" * 69)

    all_features = _get_all_features(features)
    X, y, _, _, X_test, y_test = split_training_data(df, all_features, target)

    trained_models = {}
    model_names = list(models.keys())
    n_models = len(model_names)

    oof_matrix = np.zeros((y.shape[0], n_models))
    
    for i, (k, model) in enumerate(models.items()):
        print(f"Training Model: {k}")

        if task.startswith("regression"):
            cv = skl.model_selection.KFold(
                n_splits=folds, shuffle=True, random_state=69 + i
            )
        else:
            cv = skl.model_selection.StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=69 + i
            )

        cv_models = []
        oof_pred = np.zeros(y.shape[0])

        for (train_idx, val_idx) in tqdm(cv.split(X, y), desc="training models", unit="folds"):
            X_t, X_v = X[features[k]].iloc[train_idx], X[features[k]].iloc[val_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]

            try:
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            except Exception:
                model.fit(X_t, y_t)

            if task.startswith("probability"):
                y_v_pred = model.predict_proba(X_v)[:, 1]
            else:
                y_v_pred = model.predict(X_v)

            oof_pred[val_idx] = y_v_pred
            cv_models.append(model)

        trained_models[k] = cv_models
        oof_matrix[:, i] = oof_pred

        if TargetTransformer is None:
            oof_eval = np.array(oof_pred).reshape(-1, 1)
            y_all = np.array(y).reshape(-1, 1)
        else:
            oof_eval = TargetTransformer.inverse_transform(
                np.array(oof_pred).reshape(-1, 1)
            )
            y_all = TargetTransformer.inverse_transform(
                np.array(y).reshape(-1, 1)
            )
            if verbose:
                y_t = TargetTransformer.inverse_transform(
                    np.array(y_t).reshape(-1, 1)
                )
                y_v = TargetTransformer.inverse_transform(
                    np.array(y_v).reshape(-1, 1)
                )
                y_v_pred = TargetTransformer.inverse_transform(
                    np.array(y_v_pred).reshape(-1, 1)
                )
        score = calculate_score(y_all, oof_eval, metric=task)
        print(f"Score:  {score:.4f}\n")
        print(f"***  model {k} score:  {score:.4f}  ***")

        if verbose:
            # note: plotting last fold's X_t, X_v, y_t, y_v, and corresponding preds
            plot_training_results(X_t, X_v, y_t, y_v, y_v_pred,
                task=task)

    # ---- Stacking meta-model on OOF predictions ----
    if meta_model == None:
        if task.startswith("regression"):
            meta_model = skl.linear_model.Ridge(alpha=1.0)
        else:
            meta_model = skl.linear_model.LogisticRegression()
    # note: when TargetTransformer is used, meta model is NOT trained on inverse transformed target values. 
    # Meta model output will still REQUIRE inverse transform
    tic=time()
    print(f"Selected meta model is: {meta_model}")
    print(f"Training Meta Model on {oof_matrix.shape[1]} OOF predictions and {y.shape[0]} samples")
    meta_model.fit(oof_matrix, y)
    print(f"Meta Model training completed in {time()-tic:.2f}sec")

    return trained_models, meta_model

def submit_cv_predict(X: pd.DataFrame, y: pd.DataFrame, features: dict, target:str, 
                      models: dict, task: str='regression', 
                      TargetTransformer=None, meta_model=None,
                      path: str="", file: str="sample_submission.csv", verbose: bool=True)-> pd.DataFrame:
    """
    makes predictions with cross validated models and returns predictions
    -----------
    for each model in models, makes predictions with each fold model and averages predictions for each model
    averages predictions across models to get final predictions
    returns a array of predictions for submission
    -----------
    requires: numpy, pandas, scikit learn
    optional: lightgbm, xgboost, catboost
    """
    def _plot_target(df: pd.DataFrame, target: str, title: str='predicted target distribution', hist: int=20) -> None:
        if pd.api.types.is_float_dtype(df[target]) or (df[target].dtype == int and df[target].nunique() > hist):
            sns.histplot(df[target], 
                         bins = min(df[target].nunique(), 42),  # limit number of bins for large unique value counts
                         kde = True)
        else:
            sns.countplot(data=df, x=target)
        plt.title(title)
        plt.yticks([])
        plt.show()

    if meta_model is None:
        y_test = np.zeros(y.shape[0])
    else:
        y_oof_matrix = np.zeros((y.shape[0], len(models.keys())))
    for i, (k, cv_models) in enumerate(models.items()):
        y_cv = np.zeros(y.shape[0])
        training_features = features[k]
        for model in cv_models:
            if task.startswith("probability"):
                y_cv += model.predict_proba(X[training_features])[:, 1]
            else:
                y_cv += model.predict(X[training_features])
        y_cv /= len(cv_models)
        if meta_model is None:
            y_test += y_cv
        else:
            y_oof_matrix[:, i] = y_cv

    if meta_model is None:
        y_test /= len(models.keys())
    elif task.startswith("probability"):
        y_test = meta_model.predict_proba(y_oof_matrix)[:, 1]
    else:
        y_test = meta_model.predict(y_oof_matrix)
    
    # note: when TargetTransformer is used, meta model is trained on transformed target values, 
    if TargetTransformer is None:
        y_pred = np.array(y_test).reshape(-1, 1)
    else:
        y_pred = TargetTransformer.inverse_transform(np.array(y_test).reshape(-1, 1))
    
    submission_df = pd.read_csv(f"{path}/{file}")
    submission_df[target] = y_pred
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)

    if verbose:
        _plot_target(submission_df, target, title = f"distribution of {target} predictions")

    print("=" * 6, 'save success', "=" * 6, "\n")
    print(f"Predicted target mean: {y_pred.mean():.4f} +/- {y_pred.std():.4f}")

    return submission_df

def submit_predictions(X: pd.DataFrame, y: pd.Series, target: str, 
                       models: list, task: str='regression', TargetTransformer=None, 
                       path: str="", verbose: bool=True, )-> np.ndarray:
    """
    makes predictions with trained models and saves submission file
    -----------
    for each model in models, makes predictions with the model and averages predictions
    saves predictions to submission file in path
    -----------
    requires: numpy, pandas, scikit learn
    """
    y_test = np.zeros(y.shape[0])

    for m in models:
        if task.startswith("probability"):
             y_test += m.predict_proba(X)[:, 1]
        else:
             y_test += m.predict(X) 
    y_test /= len(models)

    if TargetTransformer == None:
        y_pred = np.array(y_test).reshape(-1, 1) 
    else:
        y_pred = TargetTransformer.inverse_transform(np.array(y_test).reshape(-1, 1))

    SUBMISSION = pd.read_csv(f"{path}/sample_submission.csv")
    SUBMISSION[target] = y_pred
    SUBMISSION.to_csv('/kaggle/working/submission.csv', index=False)

    if verbose:
        plot_target_eda(SUBMISSION, target, title = f"distribution of {target} predictions")

    print("=" * 6, 'save success', "=" * 6, "\n")
    print(f"Predicted target mean: {y_pred.mean():.4f} +/- {y_pred.std():.4f}")
    return y_pred


"""
TODO: GLM feature analysis
from statsmodels.graphics.api import abline_plot
def get_glm_analysis(df, X = numeric_features, y = targets, plot = False): 
    glm_binomial = sm.GLM(df[y], df[X], family = sm.families.Binomial())
    result = glm_binomial.fit()
    print(result.summary())
    if plot == True:
        fig, ax = plt.subplots()
        resid = result.resid_deviance.copy()
        resid_std = stats.zscore(resid)
        ax.hist(resid_std, bins=33)
        ax.set_title('Histogram of standardized deviance residuals')
        plt.show()
        graphics.gofplots.qqplot(resid, line='r')
        

get_glm_analysis(TRAIN, plot = True)


TODO: Time Data
#plot auto correlation
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 

# Aggregate data by day to create the time series
df = TRAIN.groupby('date')[target].mean()

plt.figure(figsize=(10, 6))
plot_acf(df.dropna(), lags=52)
plt.title('Autocorrelation Plot')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

def decompose_plot(df):
    dtf = df.set_index('date')
    dtf = dtf[target_columns].resample('D').mean()
    dtf['year'] = dtf.index.year
    dtf['month'] = dtf.index.month

    j = len(dtf['year'].unique())
    
    fig, axs = plt.subplots(nrows= j, ncols=1, sharey = True, figsize=(15, 3 * j))

    for i, year in enumerate(dtf['year'].unique()):
        yearly_data = dtf[dtf['year'] == year]
        decomposition = seasonal_decompose(yearly_data[target], model='average', period=12)
        plt.subplot(j, 1, i + 1)
        decomposition.trend.plot()
    plt.suptitle(t = 'Trend Plots of Sales by Year', fontsize = 11) 
    plt.show()
    
    fig, axs = plt.subplots(nrows= j, ncols=1, sharey = True, figsize=(15, 3 * j))

    for i, year in enumerate(dtf['year'].unique()):
        yearly_data = dtf[dtf['year'] == year]
        decomposition = seasonal_decompose(yearly_data[target], model='average', period=12)
        plt.subplot(j, 1, i + 1)
        decomposition.seasonal.plot()
    plt.suptitle(t = 'Seasonal Plots of Sales by Year', fontsize = 11) 
    plt.show()


decompose_plot(TRAIN)


#TODO Evaluate numeric-> numeric plots
### Plot price as function of numeric features
def plot_regression(df, x, y):
    j = len(x)
    fig, axs = plt.subplots(nrows=1, ncols=j, sharey=True, figsize=(15,3))
    for i, col in enumerate(x):
        plt.subplot(1, j, i+1)
        sns.regplot(data=df, 
                    x=col, y=y,
                    x_estimator=np.mean, x_bins=50,  scatter=True, ci = 90, 
                    fit_reg=True,
                    #order = 3,
                   )
    plt.show()

def plot_pair(df, x, y):
    g = sns.PairGrid(df, y_vars=y,
                     x_vars=x, height = 5, aspect = 1,
                     )
    g.map(sns.regplot, x_estimator=np.mean)
    g.tick_params(axis = 'x', rotation =75, size = 6)
    g.tick_params(axis = 'y', size = 8)
    g.set(xlabel = None)
    plt.suptitle(t = f"Mean {y} as Function of Numeric Features", fontsize = 14, weight = 'bold') 
    sns.despine(fig=g.fig, left=True)


plot_regression(TRAIN, num_feature_names, 'price')
#plot_pair(TRAIN, num_feature_names, 'price')

"""