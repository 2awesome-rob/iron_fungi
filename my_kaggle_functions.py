#functions for use in kaggle tabular data projects
import numpy as np
import pandas as pd 

import lightgbm as lgb
import sklearn as skl
import torch

import random
import itertools

import optuna
from tqdm import tqdm
from time import time

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly as go

from multiprocessing import cpu_count


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

def summarize_data(df: pd.DataFrame, features: list)-> None:
    """prints df summary and descriptive stats for selected features"""
    print("=" * 69)
    print(df[features].info())
    print("=" * 69)
    print(df[features].head(5).T) 
    print("=" * 69)
    try:
        print(df[features].describe(include=['float', 'int']).T)
    except: pass
    try:
        non_numeric_cols = df[features].select_dtypes(include=['object', 'category', 'bool']).columns
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

def get_target_labels(df: pd.DataFrame, target: str, targets: list, cuts: int=10):
    """
    Adds target "label" columns
    Useful for visualizing numeric targets in categorical "bins"
    -----------
    if target is categorical (object or category dtype)
        - adds "label" same as target for visualization
    if target is bool or numeric with few unique values (<8)
        - adds "label" reversing the order of the target values (for better visualization)
    if target is numeric with many unique values (>=8)
        - adds "qcut_label" using pd.qcut to create quantile-based bins of the target
        - adds "cut_label" using pd.cut to create equal-width bins of the target
    returns:
    - df with new label columns added
    """
    if df[target].dtype == 'O' or df[target].dtype.name == 'category':
        df["label"] = df[target]
        print("Target is not Numeric\nLabelis target")
        #targets.append("label")
    elif df[target].nunique() < 8:
        df["label"] = df[target].max() - df[target]
        targets.append("label")
    else:
        df["qcut_label"] = cuts  - pd.qcut(df[df.target_mask.eq(True)][target], cuts, labels=False)
        df["cut_label"] = cuts  - pd.cut(df[df.target_mask.eq(True)][target], cuts, labels=False)
        df[["qcut_label", "cut_label"]] = df[["qcut_label", "cut_label"]].fillna(-1).astype('int16')
        targets.extend(["qcut_label", "cut_label"])
    return df, targets

###Data cleaning and feature engineering functions
def get_transformed_target(df: pd.DataFrame, target: str, 
                           targets: list, name: str="std",
                           TargetTransformer=skl.preprocessing.StandardScaler()
                           ):
    """
    scales or transforms targets in df with scikit learn scalers / transformers
    -----------
    returns:
    - df with transformed target
    - fitted TargetTransformer to support inverse transformation of predictions
    - updated list of targets with new transformed target column name added
    -----------
    requires: 
    pandas, scikit learn
    """
    y = df[df.target_mask.eq(True)][target].values
    TargetTransformer.fit(y.reshape(-1,1))

    y = df[target].values
    df[f"{target}_{name}"] = TargetTransformer.transform(y.reshape(-1,1))

    targets.append(f"{target}_{name}")
    return df, TargetTransformer, targets

def clean_categoricals(df: pd.DataFrame, features: list, string_length: int=3) -> pd.DataFrame:
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
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.casefold()
            .str.replace(" ", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("-", "_", regex=False)
            .str.replace(".", "_", regex=False)
            .str.replace(",", "_", regex=False)
        )
        df[col] = df[col].str[:string_length].astype('category')
    return df

def split_training_data(df: pd.DataFrame, features: list, targets, validation_size: None| float | pd.Index = None):
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
    SEED = 80085
    X_test = df[df.target_mask.eq(False)][features]
    y_test = df[df.target_mask.eq(False)][targets]

    X = df[df.target_mask.eq(True)][features]
    y = df[df.target_mask.eq(True)][targets]
    
    if type(validation_size) is float:
        X_train, X_val, y_train, y_val  = skl.model_selection.train_test_split(X, y, test_size = validation_size, random_state = SEED)
        return X_train, y_train, X_val,  y_val, X_test, y_test
    elif type(validation_size) is pd.Index: 
        X_train, y_train = X[~X.index.isin(validation_size)], y[~y.index.isin(validation_size)]
        X_val, y_val = X[X.index.isin(validation_size)], y[y.index.isin(validation_size)]
        return X_train, y_train, X_val,  y_val, X_test, y_test
    else: return X, y, X_test, y_test, X_test, y_test

def get_transformed_features(df: pd.DataFrame, features: list, FeatureTransformer):
    """
    scales or transforms features in df with scikit learn scalers / transformers
    -----------
    returns: df with transformed features
    -----------
    requires: pandas, scikit learn, tqdm
    """
    for feature in tqdm(features, desc="Transforming features", unit="features"):
        X = df[feature].values.reshape(-1,1)
        df[feature] = FeatureTransformer.fit_transform(X)
    return df    

def get_feature_interactions(df: pd.DataFrame, features:list):
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
    df = get_transformed_features(df, new_features, skl.preprocessing.PowerTransformer())
    print(f"Added {len(new_features)} inteaction features")
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
    X_features = pd.DataFrame(reduced_data, columns=cols)
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
        try:
            plot_features_eda(df[df.target_mask.eq(True)], [col_name], target, label=None)
        except:
            plot_features_eda(df, [col_name], target, label=None)
        df_sampled = df[df[f"{col_name}_noise"]==False].sample(n=min(800, df.shape[0]), random_state=69)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.scatterplot(data=df_sampled, x=df_sampled[features[0]], y=df_sampled[features[1]], 
                            hue=col_name, alpha = 0.7, palette=palette, ax=ax, legend=False)
    if not df[f"{col_name}_noise"].any():
        df.drop(columns=f"{col_name}_noise", inplace=True)
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
    if pd.api.types.is_float_dtype(df[target]) or (df[target].dtype == int and df[target].nunique() > hist):
        sns.histplot(df[target], 
                     bins = min(df[target].nunique(), 42),  # limit number of bins for large unique value counts
                     kde = True)
    else:
        sns.countplot(data=df, x=target)
    plt.title(title)
    plt.yticks([])
    plt.show()

def plot_features_eda(df: pd.DataFrame, features: list, target: str, label: str=None, 
                      sample: int=1000, y_min: float=None, y_max: float=None,
                      high_label = "Good", low_label = "Bad") -> None:
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
    ### Histogram for distribution of numeric feature (num plot 0)
    def _plot_num_distribution(ax, feature):
        sns.histplot(df[feature], ax=ax, bins = 50)
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

    ### scatterplot with trendline for numerical feature relationship to target (num plot 1)
    def _plot_num_relationship(ax, feature,  y_min=0, y_max=100):
        df_sampled = df.sample(n=min(sample, df.shape[0]), random_state=SEED)
        sns.regplot(data=df_sampled, x=feature, y=target, ax=ax,
                    scatter_kws={'alpha': 0.5, 's': 12}, line_kws={'color': 'xkcd:dusty rose', 'linestyle': "--", 'linewidth': 2})
        ax.set_title(f'{target} vs {feature}')
        ax.set_ylabel("")
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("")

    ### psedo-scatterplot with trendline for categorical feature relationship to target (cat plot 1)
    def _plot_cat_relationship(ax, feature, order, color_map, y_min=0, y_max=100):
        grouped = df.groupby(feature)
        sampled_dfs = []
        for name, group in grouped:
            frac = min(1.0, sample / len(df))
            sampled_dfs.append(group.sample(n=max(1, int(frac * len(group))), random_state=SEED))
        df_sampled = pd.concat(sampled_dfs)
        sns.stripplot(data=df_sampled, x=feature, y=target, order=order, ax=ax, zorder = 1, 
                          palette=[color_map[val] for val in order], alpha=0.5, jitter=True)
        sns.pointplot(data=df, x=feature, y=target, order=order, ax=ax, zorder = 2, 
                      color=MY_PALETTE[-1], errorbar = None)

        if len(df[target].unique()) > 5:
            for i, val in enumerate(order):
                subset = df[df[feature] == val][target].dropna()
                q25, q75 = subset.quantile([0.25, 0.75])
                ax.vlines(x=i, ymin=q25, ymax=q75, color=MY_PALETTE[-1], linewidth=2,  zorder = 3)
        
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

    ### boxplot shows outliers and limits by label  (num plot 2)
    def _plot_num_boxplot(ax, feature, label = None, top_label="", bottom_label=""):        
        if label == None:
            sns.boxplot(x = df[feature], ax=ax)
            ax.set_title(f'{feature} outliers')
        else:
            sns.boxplot(x = df[feature], palette=MY_PALETTE , ax=ax, legend = False, gap = .1,
                        hue = df[label], hue_order = sorted(df[label].dropna().unique().tolist()))
            ax.set_title(f'{feature} by target cut')
            ax.set_xlabel("")
            ax.text(df[feature].min(), -0.45, top_label, ha='left', va='center', fontsize=8, color = 'black')
            ax.text(df[feature].min(), 0.45, bottom_label, ha='left', va='center', fontsize=8, color = 'black')
        ax.set_yticks([])

    ### donut shows variation in target by category  (cat plot 2)
    def _plot_cat_donut(ax, feature, label, order, color_map, inner_label="", outer_label=""):
        cats = sorted(df[label].dropna().unique().tolist())
        ring_width = 0.7 / len(cats)
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

    ### limit number of features plotted/size of plot
    f = len(features)
    if len(features) > 20:
        print("Plotting 20 features")
        f = 20
        features = features[:20]

    ### define limits of relationship plots
    if not y_min: y_min = df[target].min()
    if not y_max: y_max = df[target].max()
    
    ### gridspec to build plot layout
    fig = plt.figure(figsize=(10, f * 3))
    gs = mpl.gridspec.GridSpec(f, 3, figure=fig, hspace=0.4)
    
    row_anchors = []
    for i, feature in enumerate(features):
        ### for each feature determine applicable plot selection
        is_cat = (df[feature].dtype == "O" or df[feature].dtype == bool or df[feature].dtype == "category" or
                  (np.issubdtype(df[feature].dtype, np.integer) and len(df[feature].dropna().unique()) < 10))
        ax0 = fig.add_subplot(gs[i, 0])
        row_anchors.append(ax0)
        if is_cat:
            order = sorted(df[feature].dropna().unique().tolist())
            color_map = get_colors(color_keys=order, get_cmap=True, n_hues=6, n_sats=5)
            _plot_cat_distribution(ax0, feature, order, color_map)
            _plot_cat_relationship(fig.add_subplot(gs[i, 1]), feature, order, color_map, y_min=y_min, y_max=y_max)
            if label != None:
                _plot_cat_donut(fig.add_subplot(gs[i, 2]), feature, label, order, color_map,
                           inner_label=low_label, outer_label=high_label)
        else:
            _plot_num_distribution(ax0, feature)
            _plot_num_relationship(fig.add_subplot(gs[i, 1]), feature, y_min=y_min, y_max=y_max)
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

### Train and evaluate
def calculate_score(actual, predicted, metric='rmse')-> float:
    """ 
    calculates score based on metric or task
    simplifies calling scikit learn metrics by allowing flexible metric names
    -----------
    returns: metric score
    for rmse use 'rmse' or 'regression'
    for accuracy use 'accuracy' or 'classification'
    for roc_auc use 'roc_auc' or 'classification_probability'
    -----------
    requires: scikit learn
    """
    if metric == 'rmse' or metric == 'regression':
        return skl.metrics.root_mean_squared_error(actual, predicted)
    elif metric == 'accuracy' or metric == 'classification':
        return skl.metrics.accuracy_score(actual, predicted)
    elif metric == 'roc_auc' or metric == 'classification_probability':
        return skl.metrics.roc_auc_score(actual, predicted)
    else:
        raise ValueError("""***UNSUPPORTED METRIC***\n
                         Supported metrics: 'rmse', 'accuracy' or 'roc_auc'""")

def plot_training_results(X_t, X_v, y_t, y_v, y_p, task: str='regression', TargetTransformer=None)-> None:
    """
    plots training results with reference model for comparison
    -----------
    fits a base model to the training data (X_t, y_t) to provide a reference point for evaluating the trained model predictions
    scores the base model predictions and trained model predictions (y_p) against validation data (y_v) 
    based on the task (regression, classification, or classification_probability)
    plots:
    for regression: actual vs predicted scatterplots for trained model and ridge regression, distribution of predictions vs actuals, and residual distribution
    for classification: confusion matrices for trained model and gaussian naive bayes, distribution of predicted probabilities
    for classification_probability: ROC curve comparing trained model and gaussian naive bayes, distribution of predicted probabilities, and confusion matrix for predicted probabilities
    -----------
    requires: pandas, scikit learn, matplotlib, numpy
    """
    if task == "regression": base_model = skl.linear_model.Ridge()
    else: base_model = skl.naive_bayes.GaussianNB()
    numeric_features = [f for f in X_t.columns.tolist() if X_t[f].dtype != "object" and X_t[f].dtype != "string"]
    base_model.fit(X_t[numeric_features], y_t)
    
    if task == "classification_probability":
        y_base = base_model.predict_proba(X_v[numeric_features])[:, 1].reshape(-1, 1)
    else:
        y_base = base_model.predict(X_v[numeric_features]).reshape(-1, 1)
        if TargetTransformer != None:
            y_base = TargetTransformer.inverse_transform(y_base).reshape(-1, 1)
    
    def plot_regression_resid(ax):
        skl.metrics.PredictionErrorDisplay.from_predictions(y_v[:1000], y_base[:1000], kind = 'actual_vs_predicted',
                                                            scatter_kwargs={"color":'xkcd:gold', "alpha":0.8},
                                                            ax = ax)
        skl.metrics.PredictionErrorDisplay.from_predictions(y_v[:1000], y_p[:1000], kind = 'actual_vs_predicted', 
                                                            scatter_kwargs={"alpha":0.8},
                                                            line_kwargs={"color":'xkcd:dusty rose'},
                                                            ax = ax)
        ax.set_title(f"Trained Model {calculate_score(y_v, y_p, metric = task):.4f} vs Ridge {calculate_score(y_v, y_base, metric = task):.4f} RMSE")

    def plot_classification_cm(ax, predictions=y_p, title = "Trained"):
        skl.metrics.ConfusionMatrixDisplay.from_predictions(y_v, predictions, cmap='bone_r', 
                                                            normalize='all', colorbar=False, ax=ax)
        ax.invert_yaxis()
        ax.set_title(f"{title} Model Accuracy {100*calculate_score(y_v, predictions, metric = 'accuracy'):.1f}%")

    def plot_classification_roc(ax):
        skl.metrics.RocCurveDisplay.from_predictions(y_v, y_p, ax=ax, name="Trained Model")
        skl.metrics.RocCurveDisplay.from_predictions(y_v, y_base, name="GaussianNB", ax=ax)
        ax.set_title("ROC Curve")

    def plot_distribution(ax):
        ax.hist(y_v, bins=min(50,2+len(np.unique(y_v))), color='xkcd:silver', alpha=0.8, density = True)
        ax.hist(y_p, bins=min(50,2+len(np.unique(y_v))), color='xkcd:ocean blue', alpha=0.9, density = True)
        ax.set_title("Prediction Distribution vs Training Distribution")
        ax.set_yticks([])
        ax.set_ylabel("Probability Density")

    def plot_residuals(ax):
        residuals = y_p - y_v
        ax.hist(residuals, bins=min(50,2+len(np.unique(residuals))), color='xkcd:dull green', alpha=0.9)
        ax.set_title("Residual Distribution")
        ax.set_yticks([])
        ax.set_ylabel("Count")

    fig = plt.figure(figsize=(9, 6))
    gs = mpl.gridspec.GridSpec(2, 3, figure=fig)

    if task == "regression":
        plot_regression_resid(fig.add_subplot(gs[:, :2]))
        plot_distribution(fig.add_subplot(gs[0, 2]))
        plot_residuals(fig.add_subplot(gs[1, 2]))
            
    elif task == "classification":
        plot_classification_cm(fig.add_subplot(gs[:, :2]))
        plot_distribution(fig.add_subplot(gs[0, 2]))
        plot_classification_cm(fig.add_subplot(gs[1, 2]), 
                               predictions = y_base,
                               title = "GaussianNB")
    
    elif task == "classification_probability":
        plot_classification_roc(fig.add_subplot(gs[:, :2]))
        plot_distribution(fig.add_subplot(gs[0, 2]))
        plot_classification_cm(fig.add_subplot(gs[1, 2]), predictions=np.round(y_p))

    plt.tight_layout()
    plt.show()

def train_and_score_model(X_train: pd.DataFrame, X_val:pd.DataFrame, 
                          y_train: pd.Series, y_val:pd.Series,
                          model, task: str="regression", 
                          verbose: bool=True, 
                          TargetTransformer=None):
    """
    trains a model and returns trained model & score
    -----------
    model: a scikit learn compatible model with fit and predict methods
    task: "regression", "classification", or "classification_probability" to determine prediction and scoring method
    returns:
    - trained model
    - score based on task
    -----------
    requires: pandas, scikit learn, numpy, matplotlib
    """
    model.fit(X_train, y_train)
    if task == "regression" or task == "classification": y_predict = model.predict(X_val)
    elif task =="classification_probability": y_predict = model.predict_proba(X_val)[:, 1]
    else: 
        print(f"Unknown task {task}")
        return model, None
    if TargetTransformer != None:
        y_v = TargetTransformer.inverse_transform(y_val.values.reshape(-1, 1))
        y_p = TargetTransformer.inverse_transform(y_predict.reshape(-1, 1))
    else:
        y_v = np.array(y_val).reshape(-1, 1)
        y_p = np.array(y_predict).reshape(-1, 1)
    score = calculate_score(y_v, y_p, metric = task)
    if verbose == True: 
        plot_training_results(X_train, X_val, y_train, y_v, y_p,
                              task=task, TargetTransformer=TargetTransformer)
    else: 
        print(f"***  model score:  {score:.4f}  ***")
    return model, score

def get_feature_importance(X_train, X_val, y_train, y_val, verbose =True, task = "regression"):
    """
    gets feature importance by training an LightGBM model
    -----------
    returns: a Series of feature importances sorted in descending order
    trains a LightGBM model on the training data and evaluates on the validation data
    -----------
    requires: pandas, lightgbm, scikit learn
    """
    if task == "regression": model = lgb.LGBMRegressor(verbose=-1)
    else: model = lgb.LGBMClassifier(verbose=-1)
    if task not in ["regression", "classification", "classification_probability"]:
        print(f"!!!  Task not recognized   !!!")
        return None
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


def study_classifier_hyperparameters(df, features, target, study_model, 
                                     metric='classification', direction='maximize',
                                     n_trials=20, timeout=1200,
                                     CORES=4, DEVICE='cpu', verbose = True):
    '''
    studies impact of hyperparameters on study models
    study_models for: 
        'lgb' : lgb.LGBMClassifier(),
        'xgb' : xgb.XGBClassifier(), 
        'catb' : catb.CatBoostClassifier(), 
        'hist' : skl.ensemble.HistGradientBoostingClassifier(),
        'rf' : skl.ensemble.RandomForestClassifier(),
        'lr' : skl.linear_model.LogisticRegression(),
    --------
    returns a dictionary of "good" hyperparamters for a given study_model, features, target
    --------
    requires: pandas, numpy, optuna, plotly.go, lgb, xgm, catb, scikit
    '''
    SEED=69
    cat_list = [f for f in features if df[f].dtype == "category"]

    def _study_objective(trial, study_model, X_train, y_train, X_val, y_val, 
                         cat_features=cat_list, metric=metric):
        if study_model == 'lgb':
            study_params = {
                'n_estimators': trial.suggest_int('n_estimators', 96, 320, step=16),  #Default=100
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0042, .42), #Default=0.1
                'num_leaves':trial.suggest_int('num_leaves', 20, 96, step=4), #Default=31
                'reg_alpha':  trial.suggest_loguniform('reg_alpha', 0.00069, .069), #Default=0.0
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.00069, .069), #Default=0.0
                'n_jobs': trial.suggest_categorical('n_jobs', [CORES]),
                'verbose': trial.suggest_categorical('verbose', [-1]),
            }
            model = lgb.LGBMClassifier(**study_params, random_state=SEED)

        elif study_model == 'xgb':
            study_params = {
                'n_estimators': trial.suggest_int('n_estimators', 96, 640, step=16),  
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0042, .42),
                'max_leaves':trial.suggest_int('max_leaves', 20, 192, step=4),
                'max_bins':trial.suggest_int('max_bins', 16, 208, step=4),
                'reg_alpha':  trial.suggest_loguniform('reg_alpha', 0.00069, .069),
                'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.00069, .069),
                'gamma': trial.suggest_float('gamma', 0.0042, 0.042),
                'device': trial.suggest_categorical('device', [DEVICE]),
                'verbosity': trial.suggest_categorical('verbosity', [0]),
                'enable_categorical': trial.suggest_categorical('enable_categorical', [True]),
            }
            model = xgb.XGBClassifier(**study_params, random_state=SEED)

        elif study_model == 'catb':
            task_type='GPU' if DEVICE == "cuda" else 'CPU'
            study_params = {
                'n_estimators': trial.suggest_int('n_estimators', 96, 320, step=16),  
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0042, 0.42),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.0001, 0.1),
                'model_size_reg': trial.suggest_loguniform('model_size_reg', 0.00042, 0.42),
                'task_type': trial.suggest_categorical('task_type', [task_type]),
                'cat_features': trial.suggest_categorical('cat_features', [cat_features]),
                'verbose': trial.suggest_categorical('verbose', [0]),
            }
            model = catb.CatBoostClassifier(**study_params, random_seed=SEED)
        
        elif study_model == 'hgb':
            study_params = {
                'max_iter': trial.suggest_int('max_iter', 1000, 1066),   #Default=100
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.0069, 0.69), #Default=0.1
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 20, 128, step=4),  #Default=31
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 16, 36), #Default=20
                'l2_regularization': trial.suggest_loguniform('l2_regularization', 0.000001, 0.01), #Default=0.0
                'early_stopping': trial.suggest_categorical('early_stopping', [True]), #Default='auto'
            }
            model = skl.ensemble.HistGradientBoostingClassifier(**study_params, random_state=SEED)
        
        elif study_model == 'rf':
            study_params = {
                'n_estimators': trial.suggest_int('n_estimators', 67, 167),  #Default=100
                'criterion': trial.suggest_categorical('criterion', ["gini", "entropy", "log_loss"]), #Default='squared_error'
                'n_jobs': trial.suggest_categorical('n_jobs', [CORES]),
                
            }
            # need work to incorporate cuml
#            if DEVICE == "cuda": model = cuml.ensemble.RandomForestClassifier(**study_params, output_type="numpy")
#            else: 
            model = skl.ensemble.RandomForestClassifier(**study_params, random_state=SEED)

        elif study_model == 'lr':
            study_params = {
                'C': trial.suggest_loguniform('C', 0.067, 6.7),  #Default=1
                'max_iter': trial.suggest_int('max_iter', 99, 291, step=24),      
                'solver': trial.suggest_categorical('solver', ["lbfgs", "sag", "saga"]), #Default='squared_error'
            }
            model = skl.linear_model.LogisticRegression(random_state=SEED)
        else:
            raise ValueError("Unrecognized model type")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val) 
        return  calculate_score(y_val, y_pred, metric=metric)

    # Main loop    
    print("=" * 69)
    print(f"study hyperparameters")
    print("=" * 69)
    
    X_train, y_train, X_val, y_val, _, _ = split_training_data(df, features, target, validation_size = 0.2)
    idx = X_train.sample(n=min(25000, X_train.shape[0]), random_state=69).index
    X_t = X_train.loc[idx]
    y_t = y_train.loc[idx]
    
    tic = time()
    study = optuna.create_study(direction=direction)
    study.optimize(lambda trial: _study_objective(trial, study_model, X_t, y_t, X_val, y_val),
                   n_trials=n_trials,
                   timeout=timeout)
    toc = time()
    trial = study.best_trial
    if verbose == True:
        fig = optuna.visualization.plot_optimization_history(study)
        go.io.show(fig)
        fig = optuna.visualization.plot_slice(study)
        go.io.show(fig)
        print("=" * 69)
        print(study.best_params)
        print("=" * 69)
        print(f"Average Trial Duration {(toc - tic)/n_trials:.3f}sec")
        print(f"Number of finished trials: {len(study.trials)}") 
    print(f"Best trial:\n  {metric} value: {trial.value}\n  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    return study.best_params


def submit_predictions(X: pd.DataFrame, y: pd.Series, target: str, models: list, 
                     task: str='regression', path: str="", 
                     verbose: bool=True, TargetTransformer=None)-> np.ndarray:
    
    SUBMISSION = pd.read_csv(f"{path}/sample_submission.csv")
    y_test = np.zeros(X.shape[0])
    
    for m in models:
        if task == "classification_probability":
             y_test += m.predict_proba(X)[:, 1]
        else:
             y_test += m.predict(X) 
    y_test /= len(models)

    if TargetTransformer == None:
        y_pred = np.array(y_test).reshape(-1, 1) 
    else:
        y_pred = TargetTransformer.inverse_transform(np.array(y_test).reshape(-1, 1))

    SUBMISSION[target] = y_pred
    SUBMISSION.to_csv('/kaggle/working/submission.csv', index=False)

    if verbose:
        plot_target_eda(SUBMISSION, target, title = f"distribution of {target} predictions")

    print("=" * 6, 'save success', "=" * 6, "\n")
    print(f"Predicted target mean: {y_pred.mean():.4f} +/- {y_pred.std():.4f}")
    return y_pred