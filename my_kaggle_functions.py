################################################################################
# Bob's functions for use in KAGGLE TABULAR DATA PROJECTS
#
# target_mask: dataframe column to slice training(True)/testing(False) data
# target: generally the dataframe column with the predicted feature
# targets: list of dataframe columns with non-predictive features
#           includeing target, additional labels and masks
# features: list of dataframe columns with predictive features
#
#
# TASK = 'regression', 'classification' or 'probability' + a sci-kit metric
# DEVICE = 'cpu' or 'cuda'
# CORES = # CPU cores to use
#
################################################################################

# Import required libraries and toolkits 
import numpy as np
import pandas as pd 

import torch
import torch.nn as nn

import sklearn as skl
import lightgbm as lgb
import xgboost as xgb
import catboost as catb

try:
    import umap
except Exception:
    print(f"UMAP emeddings not available in this environment")

from scipy import stats
import math
import random
#import statsmodels.api as sm
#import statsmodels.formula.api as smf

import optuna
import itertools
from tqdm import tqdm
from time import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as go

from multiprocessing import cpu_count
from typing import List, Dict, Optional, Tuple, Union
from typing import NamedTuple

########################################
# Initialization and data loading functions
def _get_cmap(default:bool=True):
    """helper function to maintain color consistency"""
    if default: return 'cividis'
    else: return 'viridis'

def _get_colors(color_keys: List[str] = None, n_hues: int = 5, n_sats: int = 5
                ) -> Union[List, Dict]:
    """
    Generate color palettes for consistent visualizations.
    If color_keys is None, returns a palette (list of colors).
    If color_keys is provided, returns a dict mapping keys to colors.
    """
    my_palette = sns.xkcd_palette([
        'ocean blue', 'gold', 'dull green', 'dusty rose', 'dark lavender',
        'carolina blue', 'sunflower', 'lichen', 'blush pink', 'dusty lavender'])
    if color_keys is None:
        return my_palette

    n_colors = len(color_keys)
    if n_colors <= len(my_palette):
        palette = my_palette[:n_colors]

    elif n_colors <= n_hues * n_sats:
        palette = []
        for j in range(n_hues):
            for i in range(n_sats):
                palette.append(sns.desaturate(my_palette[j % len(my_palette)], 1 - .2 * i))
        palette = palette[:n_colors]

    else:
        cmap_name = _get_cmap()
        cmap = mpl.colormaps[cmap_name].resampled(n_colors)
        palette = [cmap(i / n_colors) for i in range(n_colors)]
        
    c_map = dict(zip(color_keys, palette))
    #always plot noise in grey
    if type(color_keys[0])==str:
        c_map['noise'] = 'xkcd:silver'
    else:
        c_map[-1] = 'xkcd:silver'
    return c_map

class Globals(NamedTuple):
    device: str
    cores: int

def set_globals(seed: int = 67, verbose: bool = True) -> Globals:
    """
    Set global variables and configurations for the project.
    Returns a Globals namedtuple: (device, cores)
    """
    # Visualization settings
    pd.set_option('display.max_rows', 25)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_colwidth', 15)
    pd.set_option('display.width', 130)
    pd.set_option('display.precision', 4)

    # Custom seaborn/matplotlib style
    my_palette = _get_colors()
    sns.set_theme(context='paper',
                  style='ticks',
                  palette=my_palette,
                  rc={"figure.figsize": (9, 3),
                      "axes.spines.right": False,
                      "axes.spines.top": False})

    # Random seed settings for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Device settings
    try:
        torch.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = 'cpu'
    cores = min(4, cpu_count())

    if verbose:
        print(f"Using device: {device}")
        print(f"Using {cores} CPU cores when multiprocessing")

    return Globals(device, cores)

DEVICE, CORES = set_globals(verbose=False)

class TabularData(NamedTuple):
    df: 'pd.DataFrame'
    features: List[str]
    targets: List[str]
    target: str

def _clean_feature_names(features: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """helper function to clean feature names"""
    clean_map = {col: col.casefold().strip().
                 replace(" ","_").
                 replace("(","_").
                 replace(")","").
                 replace("-","_").
                 replace(",","_").
                 replace(".","_") for col in features}
    return [clean_map[col] for col in features], clean_map

def load_tabular_data(
    path: str, extra_data: Optional[str] = None, rename_col: Optional[Dict[str, str]] = None,
    id_feature: Optional[List[str]] = None, csv_sep: str = ",", verbose: bool = True
    ) -> TabularData:
    """
    Reads Kaggle type tabular data from csv files into a single DataFrame.
    Returns a namedtuple: (df, features, targets, target)
    """
    # Read CSVs
    df_train = pd.read_csv(f"{path}/train.csv", sep=csv_sep)
    df_test = pd.read_csv(f"{path}/test.csv", sep=csv_sep)
    df_submission = pd.read_csv(f"{path}/sample_submission.csv", sep=csv_sep)

    targets = list(df_submission.columns)
    features = list(df_test.columns)
    if id_feature is None:
        id_feature = [feature for feature in features if feature in targets]
    if len(id_feature) != 1:
        raise ValueError(f"Expected exactly one ID column: id_feature = {id_feature}")
    targets = [f for f in targets if f not in id_feature]
    features = [f for f in features if f not in id_feature]
    target = targets[0]

    # Merge and Concat
    df_test = df_test.merge(df_submission, how='left', on=id_feature)
    df = pd.concat([df_train.assign(target_mask=True),
                    df_test.assign(target_mask=False)],
                    ignore_index=True)

    # Optionally add extra training data
    if extra_data:
        try:
            df_extra = pd.read_csv(extra_data, sep=csv_sep)
            if df_extra.shape[1] == 1:
                alt_sep = ";" if csv_sep == "," else ","
                df_extra = pd.read_csv(extra_data, sep=alt_sep)
            if rename_col:
                df_extra.rename(columns=rename_col, inplace=True)
            
            missing_cols = set(targets + features + id_feature) - set(df_extra.columns)
            if missing_cols:
                print(f"Extra Data missing columns: {missing_cols}")
            extra_cols = [f for f in df_extra.columns if f not in df_columns]
            if extra_cols:
                print(f"Extra Data has additional columns: {extra_cols}")

            df_extra[id_feature[0]] = range(len(df), len(df) + len(df_extra))
            overlap = set(df[id_feature[0]]).intersection(df_extra[id_feature[0]])
            if overlap:
                raise ValueError(f"*** ERROR Duplicate IDs generated in extra_data:\n {overlap} ***")
            df = pd.concat([df.assign(extra_mask=False), 
                            df_extra.assign(extra_mask=True, target_mask=True)],
                           ignore_index=True)
        except Exception as e:
            print(f"*** ERROR loading extra_data: {e} ***")

    # Clean feature names
    features, clean_map = _clean_feature_names(features)
    df.rename(columns=clean_map, inplace=True)

    # Add target_label
    cuts = 5
    if (df[target].dtype in [int, float]) and df[target].nunique() > cuts:
        df["target_label"] = pd.qcut(df[df.target_mask.eq(True)][target], cuts, labels=False)
        df["target_label"] = df["target_label"].fillna(-1).astype('int16').astype('category')
    else:
        df["target_label"] = df[target]

    if verbose:
        print("=" * 69)
        print(f"Loaded {df.target_mask.eq(True).sum()} training samples of {len(features)} features and {len(targets)} target(s).")
        print(f"Loaded {df.target_mask.eq(False).sum()} testing samples.")
        print(f"DataFrame shape: {df.shape}. Ready to explore, engineer, and predict!")
        print("=" * 69)

    targets += ["target_label", "target_mask"]
    if extra_data: targets.append("extra_mask")
    df.set_index(id_feature, inplace=True)
    return TabularData(df, features, targets, target)

def summarize_data(df: pd.DataFrame, features: List[str]) -> None:
    """
    Print DataFrame summary and descriptive stats for selected features.
    """
    print("=" * 69)
    if 'target_mask' in df.columns:
            df_plot = df[df.target_mask.eq(True)][features]
    else:
        df_plot = df[features]
    print("=" * 69)
    print(df_plot[features].info())
    print("")

    print("=" * 69)
    print(df_plot.head(5).T)
    print("")
   
    numeric_cols = df_plot.select_dtypes(include=['float', 'int']).columns
    non_numeric_cols = df_plot.select_dtypes(include=['object', 'category', 'bool']).columns

    if len(numeric_cols) > 0:
        print("=" * 69)
        print(df_plot[numeric_cols].describe().T)
   
    if len(non_numeric_cols) > 0:
        print("=" * 69)
        print(df_plot[non_numeric_cols].describe().T)

    return

########################################
# EDA functions
def plot_target_eda(df: pd.DataFrame, target: str,
                     title: str = 'Target Distribution') -> None:
    """
    Plot the distribution of the target variable.
    - For continuous targets: histogram with KDE.
    - For categorical targets: sorted countplot.
    """
    hist_threshold = 10  # Use histplot for high cardinality integers
    if 'target_mask' in df.columns:
        df_plot = df[df.target_mask.eq(True)][target].to_frame()
    else:
        df_plot = df[[target]].copy()

    n_unique = df_plot[target].nunique()

    if df_plot[target].dtype == 'float' or (
        df_plot[target].dtype == 'int' and n_unique > hist_threshold):
        sns.histplot(df_plot[target], kde=True)
    else:
        df_plot[target] = df_plot[target].astype(str).astype('category')
        tgt_order = sorted(df_plot[target].unique().tolist())
        df_plot[target] = pd.Categorical(df_plot[target], categories=tgt_order, ordered=True)
        sns.countplot(data=df_plot, x=target, order=tgt_order)
    plt.title(title)
    plt.yticks([])
    plt.tight_layout()
    plt.show()

def plot_features_eda(df: pd.DataFrame, features: List[str], target: str, 
    target_label: str = "target_label", high_label: str = "", low_label: str = "", 
    y_min: float = None, y_max: float = None
    ) -> None:
    """
    Visual EDA for a list of features against a target.
    For each feature, plots:
        - Distribution (histogram/countplot/lineplot)
        - Target relationship (scatter, density, or strip/point)
        - Outlier/boxplot or donut/categorical breakdown
    """
    if 'target_mask' in df.columns:
        df_plot = df[df.target_mask.eq(True)].copy()
    else:
        df_plot = df.copy()

    # declare constants and null interators
    MY_PALETTE = _get_colors()
    SAMPLE = min(1000, df_plot.shape[0])
    SEED = 67
    row_anchors = []
    unplotted = []
    
    # single precalculation of sampled/sorted data
    label_order = sorted(df_plot[target_label].dropna().unique().tolist(), reverse=True)
    df_plot[target_label] = pd.Categorical(df_plot[target_label], categories=label_order, ordered=True)
    df_scatter = df_plot.sample(n=SAMPLE, random_state=SEED)
    df_line = df_plot.sample(n=SAMPLE//10, random_state=SEED)

    # determine nature of target
    tgt_cat = (df_plot[target].dtype == "O" or df_plot[target].dtype == bool or 
               df_plot[target].dtype == "category" or df_plot[target].nunique() < 4)
    if tgt_cat:
        df_plot[target] = df_plot[target].astype(str).astype('category')
        tgt_order = sorted(df_plot[target].dropna().unique().tolist(), reverse=True)
        df_plot[target] = pd.Categorical(df_plot[target], categories=tgt_order, ordered=True)
    else:
        if not y_min: y_min = df_plot[target].min()
        if not y_max: y_max = df_plot[target].max()

    if len(features) > 20: print("Plotting first 20 features")
    features = features[:20]
    #gridspec to build plot layout
    fig = plt.figure(figsize=(10, len(features) * 3))
    gs = mpl.gridspec.GridSpec(len(features), 3, figure=fig, hspace=0.4)

    def _plot_num_distribution(ax, feature):
        sns.histplot(df_plot[feature], ax=ax, discrete=True)
        ax.set_title(f'{feature} distribution')
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.set_yticks([])

    def _plot_num_relationship(ax, feature, y_min=y_min, y_max=y_max):
        sns.regplot(data=df_scatter, x=feature, y=target, ax=ax,
            scatter_kws={'alpha': 0.5, 's': 12},
            line_kws={'color': 'xkcd:rust', 'linestyle': ":", 'linewidth': 1})
        sns.lineplot(data=df_line, x=feature, y=target, ax=ax,
            color='xkcd:rust', linewidth=1)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f'{target} vs {feature}')
        ax.set_xlabel("")
        ax.set_ylabel("")

    def _plot_num_tgt_cat_relationship(ax, feature):
        sns.histplot(data=df_plot, stat='percent', x=feature, y=target,
                     discrete=[True, True], legend=False, ax=ax,
                     pthresh=0.002, pmax=0.998, zorder=0)
        quadmesh = ax.collections[0]
        densities = quadmesh.get_array().data
        densities_flat = densities.ravel()
        x_edges = quadmesh._coordinates[0, :, 0]
        y_edges = quadmesh._coordinates[:, 0, 1]
        ny = len(y_edges) - 1
        nx = len(x_edges) - 1
        sorted_idx = np.argsort(densities_flat)
        N = min(len(x_edges)//2, 5)
        top_idx = sorted_idx[-N:]
        bottom_idx = sorted_idx[:N]

        def _annotate_bins(indices, symbol, color='xkcd:rust'):
            for idx in indices:
                iy, ix = np.unravel_index(idx, (ny, nx))
                x_center = (x_edges[ix] + x_edges[ix+1]) / 2
                y_center = (y_edges[iy] + y_edges[iy+1]) / 2
                ax.text(float(x_center), float(y_center), symbol,
                        ha="center", va="center",
                        color=color, fontsize=10, zorder=2)
        
        _annotate_bins(top_idx, "●")
        _annotate_bins(bottom_idx, "⊥")
        ax.set_title(f'{target} vs {feature}')
        ax.set_xlabel("")
        ax.set_ylabel("")
        y_tic = ax.get_yticks()
        y_lbl = ["" for _ in y_tic]
        if len(y_lbl) > 0:
            y_lbl[0] = label_order[0]
            y_lbl[-1] = label_order[-1]
        ax.set_yticks(y_tic, y_lbl, rotation=90)

    def _plot_num_boxplot(ax, feature, top_label=high_label, bottom_label=low_label):
        sns.boxplot(x=df_plot[feature], palette=MY_PALETTE , ax=ax, legend=False, gap=0.1,
                    hue=df_plot[target_label], hue_order=label_order)
        if top_label == "" and bottom_label == "":
            top_label, bottom_label = label_order[0], label_order[-1]
        ax.text(df_plot[feature].min(), -0.45, top_label, ha='left', va='center', fontsize=8, color='xkcd:steel grey')
        ax.text(df_plot[feature].min(), 0.45, bottom_label, ha='left', va='center', fontsize=8, color='xkcd:steel grey')
        ax.set_title(f'{feature} outliers by target')
        ax.set_xlabel("")
        ax.set_yticks([])

    def _plot_cat_distribution(ax, feature, order, color_map):
        sns.countplot(data=df_plot, x=feature, order=order,
                      palette=[color_map[val] for val in order], ax=ax)
        ax.set_title(f'{feature} distribution')
        ax.set_xlabel("")
        if len(order) >= 8:
            x_tic = ax.get_xticks()
            if len(order) >= 20:
                x_lbl = [s if i % 5 == 0 else "" for i, s in enumerate(order)]
            else: x_lbl = order
            ax.set_xticks(x_tic, x_lbl, rotation=90)
        ax.set_ylabel("Count")
        ax.set_yticks([])

    def _plot_cat_relationship(ax, feature, order, color_map, y_min=y_min, y_max=y_max):
        grouped = df_plot.groupby(feature)
        sampled_dfs = []
        for name, group in grouped:
            if len(group) == 0:
                continue
            n_samples = max(1, int(SAMPLE * len(group) / len(df_plot)))
            sampled_dfs.append(group.sample(n=n_samples, random_state=SEED))
        if sampled_dfs:
            df_sampled = pd.concat(sampled_dfs)
            sns.stripplot(data=df_sampled, x=feature, y=target, order=order, ax=ax, zorder = 1, 
                              palette=[color_map[val] for val in order], alpha=0.5, jitter=True)

        sns.pointplot(data=df_plot, x=feature, y=target, order=order, ax=ax, zorder = 2, 
                      color='xkcd:rust', errorbar = None)

        if len(df_plot[target].unique()) > 5 and len(order) < 20:
            for i, val in enumerate(order):
                subset = df_plot[df_plot[feature] == val][target].dropna()
                q25, q75 = subset.quantile([0.25, 0.75])
                ax.vlines(x=i, ymin=q25, ymax=q75, color='xkcd:rust', linewidth=1.5,  zorder = 3)
        
        ax.set_title(f'{target} vs {feature}')
        ax.set_xlabel("")
        if len(order) >= 8:
            x_tic = ax.get_xticks()
            if len(order) >= 20:
                x_lbl = [s if i % 5 == 0 else "" for i, s in enumerate(order)]
            else: x_lbl = order
            ax.set_xticks(x_tic, x_lbl, rotation=90)
        ax.set_ylabel("")
        ax.set_ylim(y_min, y_max)

    def _plot_cat_tgt_cat_relationship(ax, feature, order):
        sns.histplot(data=df_plot, stat='percent', x=feature, y=target_label, zorder=0, 
                                  legend=False, discrete=[True,True], ax=ax)

        if len(order) < 20:
            mode_points = df_plot.groupby(feature)[target_label].agg(lambda x: x.value_counts().idxmax()).reset_index()
            sns.pointplot(data=mode_points, x=feature, y=target_label, zorder=1,
                                        color='xkcd:rust', errorbar=None)
        ax.set_title(f'{target} vs {feature}')
        ax.set_xlabel("")
        if len(order) >= 8:
            x_tic = ax.get_xticks()
            if len(order) >= 20:
                x_lbl = [s if i % 5 == 0 else "" for i, s in enumerate(order)]
            else: x_lbl = order
            ax.set_xticks(x_tic, x_lbl, rotation=90)
        ax.set_ylabel("")
        y_tic = ax.get_yticks()
        y_lbl = ["" for _ in y_tic]
        if len(y_lbl) > 0:
            y_lbl[0] = label_order[0]
            y_lbl[-1] = label_order[-1]
        ax.set_yticks(y_tic, y_lbl, rotation=90)

    def _plot_cat_donut(ax, feature, order, color_map, outer_label=high_label, inner_label=low_label):
        ring_width = 0.7 / len(label_order)
        if inner_label == "" and outer_label =="":
            outer_label, inner_label  = label_order[0], label_order[-1]
        for i, label in enumerate(label_order):
            value_counts = df[df[target_label] == label][feature].value_counts()
            sorted_counts = value_counts.reindex(order).dropna()
            if len(order) > 20:
                w_lbl = [s if i % 5 == 0 else "" for i, s in enumerate(sorted_counts.index)]
            else: w_lbl = sorted_counts.index
            slice_colors = [color_map[val] for val in sorted_counts.index]
            radius = 1 - ring_width * i
            ax.pie(sorted_counts, radius=radius, colors=slice_colors,
                   wedgeprops=dict(width=ring_width, edgecolor='w'),
                   labels=w_lbl if i == 0 else None)
            ax.set_title(f'{feature} pct by target')
            ax.text(0, 0, inner_label, ha='center', va='center', fontsize=8, color='xkcd:steel grey')
            ax.text(-1.3, 1.15, outer_label, ha='left', va='center', fontsize=8, color='xkcd:steel grey')

    def _plot_dt_distribution(ax, feature, freq, window):
        df_td = df_plot.groupby([pd.Grouper(key=feature, freq=freq)])[target].count().rename("count_td").reset_index()

        sns.lineplot(data=df_td, x=feature, y="count_td", ax=ax)
        ax.set_title(f'{target} {window} distribution')
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.set_yticks([])
       
    def _plot_dt_relationship(ax, feature, freq, window, y_min=y_min, y_max=y_max):
        df_td = df_plot.groupby([pd.Grouper(key=feature, freq=freq)])[target].mean().rename("mean_td").reset_index()

        sns.scatterplot(data=df_scatter, x=feature, y=target, ax=ax, zorder=0, alpha=0.5)
        sns.lineplot(data=df_td, x=feature, y="mean_td", ax=ax, zorder=1, color='xkcd:rust')
        ax.set_title(f'{target} {window} mean')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(y_min, y_max)

    def _plot_dt_tgt_cat_relationship(ax, feature, freq, window):
        for label in label_order:
            df_td = df.loc[df[target] == label].groupby(
                pd.Grouper(key=feature, freq=freq)
                )[target].count().rename(f"count_{label}_td").reset_index()
            sns.lineplot(data=df_td, x=feature, y=f"count_{label}_td", ax=ax)
        ax.set_title(f'{target} {window} count')
        ax.set_xlabel("")
        ax.set_ylabel("")

    def _plot_dt_autoplot(ax, feature):
        #TODO should we do any sampling prior to plotting?
        ds = df[[feature, target]].set_index(feature)
        pd.plotting.autocorrelation_plot(ds[target], ax=ax)
        ax.set_title(f'{target} autocorrelation by {feature}')
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    # MAIN PLOT LOOP
    for i, feature in enumerate(features):
        is_num = df_plot[feature].dtype == 'float' or (df_plot[feature].dtype=='int' and df_plot[feature].nunique() > 4)
        if is_num:
            ax0 = fig.add_subplot(gs[i, 0])
            row_anchors.append(ax0)
            _plot_num_distribution(ax0, feature)
            _plot_num_boxplot(fig.add_subplot(gs[i, 2]), feature)
            if tgt_cat: 
                _plot_num_tgt_cat_relationship(fig.add_subplot(gs[i, 1]), feature)
            else: 
                _plot_num_relationship(fig.add_subplot(gs[i, 1]), feature)
            continue

        if df_plot[feature].dtype == 'O':
            try: df_plot[feature] = pd.to_datetime(df_plot[feature])
            except Exception: pass

        is_dt = True if pd.api.types.is_datetime64_any_dtype(df_plot[feature]) else False
        if is_dt:
            if (df_plot[feature].max() - df_plot[feature].min()) > pd.Timedelta(days=365):
                freq, window ="MS", "monthly"
            else:
                freq, window = "W", "weekly"
            #TODO - expand to higher freq/smaller window if required
            ax0 = fig.add_subplot(gs[i, 0])
            row_anchors.append(ax0)
            _plot_dt_distribution(ax0, feature, freq, window)
            _plot_dt_autoplot(fig.add_subplot(gs[i, 2]), feature)
            if tgt_cat: 
                _plot_dt_tgt_cat_relationship(fig.add_subplot(gs[i, 1]), feature, freq, window)
            else: 
                _plot_dt_relationship(fig.add_subplot(gs[i, 1]), feature, freq, window)
            continue

        is_cat = (df_plot[feature].dtype == "O" or 
                  df_plot[feature].dtype == bool or 
                  df_plot[feature].dtype == "category" or 
                  (df_plot[feature].dtype=='int' and df_plot[feature].nunique() <= 4))
        if is_cat:
            order = sorted(df_plot[feature].dropna().unique().tolist())
            if len(order) <= 64:
                df_plot[feature] = pd.Categorical(df_plot[feature], categories=order, ordered=True)
                color_map = _get_colors(color_keys=order, n_hues=6, n_sats=5)
                ax0 = fig.add_subplot(gs[i, 0])
                row_anchors.append(ax0)
                _plot_cat_distribution(ax0, feature, order, color_map)
                _plot_cat_donut(fig.add_subplot(gs[i, 2]), feature, order, color_map)
                if tgt_cat: 
                    _plot_cat_tgt_cat_relationship(fig.add_subplot(gs[i, 1]), feature, order)
                else: 
                    _plot_cat_relationship(fig.add_subplot(gs[i, 1]), feature, order, color_map)
                continue
            else:
                unplotted.append(feature)
            #TODO: alternate plot when high ordinal >64 categorical?
            # perhaps we recall ax0 and merge across all plot column windows?
            #else:
            #    ax0 = fig.add_subplot(gs[i, :2])
            #    _plot_cat_distribution(ax0, feature, order, color_map)
        else:
            unplotted.append(feature)

    if unplotted: print(f"Unplotted Features: {unplotted}")
    # add tear lines between features
    for i in range(len(features) - 1):
        bottom_y = row_anchors[i].get_position().y0
        top_y = row_anchors[i + 1].get_position().y1
        y_pos = (bottom_y + top_y) / 2
        line = mpl.lines.Line2D([0.05, 0.95], [y_pos, y_pos], transform=fig.transFigure,
                      color="xkcd:slate", linewidth=0.5, linestyle='--')
        fig.add_artist(line)
    plt.show()

def plot_compare_features(df: pd.DataFrame, features: List[str], target:Optional[str] = None,
                  sample: int=250, title: str="Feature to Feature Comparison") -> None:
    """
    for visualizing feature to feature comparisons
    -----------
    if few numeric features plots:
        - pairwise scatterplots for numeric features
        - kde histograms on the diagonal
        - contour lines on the lower triangle to show density of points in scatterplots
    if many numeric features, plots:
        - heatmap of linear correlation between features
    for performance, limits scatterplots to a sample of the data
    """
    if 'target_mask' in df.columns:
        df_plot = df[df.target_mask.eq(True)]
    else:
        df_plot = df.copy()

    plot_features = [f for f in features if
                     (df[f].dtype == 'float' or df[f].dtype == 'int')]
    kwargs = {'hue': None}

    if target is not None:
        if df_plot[target].nunique() < 8:
            kwargs['hue'] = target
            plot_features = [target] + plot_features
        elif (df_plot[target].dtype=='int' or df[target].dtype == 'float'):
            plot_features = [target] + plot_features

    if len(plot_features) < 10:
        df_plot = df_plot.sample(n = min(sample, df_plot.shape[0]), random_state=69)    
        g = sns.pairplot(df_plot[plot_features], diag_kind="kde", **kwargs)
        g.map_lower(sns.kdeplot, levels=4, color='xkcd:steel grey')
        g.figure.suptitle(title, x = 0.98, ha = 'right', y=1.01)
    else:
        plt.figure(figsize=(8,8))
        cmap = _get_cmap() 
        sns.heatmap(data=df[plot_features].corr(), 
            mask=np.tril(df[plot_features].corr()), 
            annot=True if (len(plot_features)<16) else False, 
            fmt='.2f', 
            square=True,
            cbar=False,
            cmap=cmap, 
            vmin = -1, vmax = 1,
            linewidth=0.5, linecolor='white')
        plt.xlabel("")
        plt.ylabel("")
    plt.show()

def print_pca_loadings(df: pd.DataFrame, features: List[str], filter_small: bool=True) -> None:
    """
    normalizes and prints PCA loadings for selected features 
    -----------
    useful for understanding relationships in the data and informing feature engineering decisions
    -----------
    requires: pandas, scikit learn
    """ 
    plot_features = [f for f in features if
                     (df[f].dtype == 'float' or df[f].dtype == 'int')]

    X = df[df.target_mask.eq(True)][plot_features[:10]]
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

def plot_feature_transforms(df: pd.DataFrame, features: List[str], Transformer=None)-> None:
    """
    Plots histogram distribution of feature variable with different transformations 
    Informs feature transformation and scaling decisions 
    """
    for feature in features:
        if 'target_mask' in df.columns:
            df_plot = df[df.target_mask.eq(True)][feature].to_frame()
        else:
            df_plot = df[feature].to_frame()
        X = df_plot[feature].values.reshape(-1,1)
        df_plot['StandardScaler']  = skl.preprocessing.StandardScaler().fit_transform(X)
        df_plot['PowerTransformer']  = skl.preprocessing.PowerTransformer().fit_transform(X)
        df_plot['QuantileTransformer']  = skl.preprocessing.QuantileTransformer().fit_transform(X)
        df_plot['MinMaxScaler'] = skl.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
        if Transformer is None:
            df_plot['np_log1p'] = np.log1p(X)
        else:
            df_plot['Transformer'] = Transformer.fit_transform(X)
        columns = list(df_plot.columns)
        fig, axs = plt.subplots(nrows=1, ncols=len(columns), sharey=True, figsize=(15,3))
        for i, col in enumerate(columns):
            plt.subplot(1, len(columns), i+1)
            sns.histplot(data=df_plot, stat='percent', x=col, kde=False, bins=30, legend = False)
        plt.show()

def plot_countplots(df: pd.DataFrame, features: List[str]) -> None:
    """
    countplot for dataset to dataset comparisons
    -----------
    plots number of unique values for each feature by data set (train, test, original)
    """
    print("=" * 69)
    datasets = 3 if 'extra_mask' in df.columns else 2

    df_train = df[df.target_mask.eq(True)]
    df_test = df[df.target_mask.eq(False)]

    my_blues = sns.dark_palette("SteelBlue", n_colors=len(features)*2, reverse = True)
    my_yellows = sns.light_palette("GoldenRod", n_colors=len(features)*2, reverse = True)
    
    fig, axs = plt.subplots(figsize=(12,4))

    if datasets == 3:
        my_greys = sns.light_palette("LightGrey", n_colors=len(features))
        df_extra = df_train[df_train.extra_mask.eq(True)]
        df_train = df_train[df_train.extra_mask.eq(False)]
        sns.barplot(x = features, y = df_extra[features].nunique().values, palette = my_greys, linewidth = 1.5, edgecolor = 'k')

    sns.barplot(x = features,
                y = df_train[features].nunique().values, 
                palette = my_blues, 
                gap = 0.12,
                linewidth = 1, edgecolor = 'k')
    sns.barplot(x = features,
                y = df_test[features].nunique().values, 
                palette = my_yellows, 
                gap = 0.36,
                linewidth = 1, edgecolor = 'k')

    j = len(axs.patches) / datasets
    for i, p in enumerate(axs.patches):
        value = f"{p.get_height():,.0f}"
        y = p.get_height()
        if datasets == 3 and i < j:
            x = p.get_x() 
            axs.text(x, y, value, fontsize=9, ha='left', va='center',                 #y position shifted to minimize overlap and enable reading numbers behind
                    bbox=dict(facecolor="LightGrey", boxstyle='round', linewidth=1, edgecolor='k'))
        elif (datasets == 3 and i < 2 * j) or (datasets == 2 and i < j):
            x = p.get_x() + 0.5 * p.get_width()
            axs.text(x, y, value, fontsize=9, ha='center', va='center', color = 'w',                 #y position shifted to minimize overlap and enable reading numbers behind
                    bbox=dict(facecolor="SteelBlue", boxstyle='round', linewidth=1, edgecolor='k'))
        else:
            x = p.get_x() + 1.2 * p.get_width()
            axs.text(x, y, value, fontsize=9, ha='right', va='center',                 #y position shifted to minimize overlap and enable reading numbers behind
                    bbox=dict(facecolor='GoldenRod', boxstyle='round', linewidth=1, edgecolor='k'))
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(())
    t = 'feature counts in original (grey), training (blue), and test datasets(yellow)' if datasets == 3 else 'feature counts in training (blue), and test datasets(yellow)'
    plt.suptitle(t = t, fontsize = 10, x = 0.9, ha ='right') 
    sns.despine()
    plt.show()

def check_categoricals(df: pd.DataFrame, features: list, pct_diff: float=0.1, verbose: bool=True)-> None:
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
    
    def _print_consistency(feature, feature_values):
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

    if verbose:
        plot_countplots(df, features)

def check_all_features_scaled(df: pd.DataFrame, targets:list)-> None:
    obj_features = [f for f in df.columns if f not in targets and
                        df[f].dtype == 'O']
    if obj_features != []:
        print(f"Object features remain in training set: {obj_features}")
        
    num_features = [f for f in df.columns if f not in targets and 
                (df[f].dtype == 'float' or df[f].dtype == 'int')]
    if num_features==[]:
        print("No numeric features in DataFrame")
        return
    
    skewed_features = [f for f in num_features if (df[f].mean() > 1 or df[f].mean() <-1)]
    unscaled_features = [f for f in num_features if (df[f].max() > 10 or df[f].min() <-10)]
    if skewed_features == [] and unscaled_features == []:
        print("All features scaled")
    else:
        if skewed_features:
            print(f"Consider transforming: {skewed_features}")
        if unscaled_features:
            print(f"Consider scaling: {unscaled_features}")
    return        

def plot_lag(df:pd.DataFrame, time_feature:str, target:str, lag: int=5, sample: int=5000):
    """
    plots lag and autocorrelation on a downsampe of the df
    needs a better sampling technique
    """
    print("Experimental - need to validate")
    if 'target_mask' in df.columns:
        df_plot = df[df.target_mask.eq(True)]
        df_plot = df_plot.sample(min(len(df), sample))
    else:
        df_plot = df.sample(min(len(df), sample))

    ds = df_plot[[time_feature, target]].set_index(time_feature)
    _, axs = plt.subplots(nrows=2, ncols=1, figsize=(5, 3))
    pd.plotting.autocorrelation_plot(ds, ax=axs[0])
    pd.plotting.lag_plot(ds, lag=lag, ax=axs[1])
    plt.show()

########################################
# Data cleaning 
def check_duplicates(df: pd.DataFrame, features: List[str], target: str,
                      drop: bool=False, verbose: bool=True) -> pd.DataFrame:
    """
    Checks for duplicate rows of selected features in df.
    Returns:
    - DataFrame with duplicate rows dropped from training data if drop=True.
    """
    if 'target_mask' in df.columns:
        mask = df.target_mask.eq(True)
        train_df = df[mask]
        test_df = df[~mask]
    else:
        print("Unable to separate test and training data")
        print(f"{df[features].duplicated(keep=False).sum()} duplicates in DataFrame")
        return df

    # Find duplicates in training data
    duplicate_mask = train_df[features].duplicated(keep=False)
    n_duplicates = duplicate_mask.sum()

    # Find overlap between train and test
    overlap = pd.merge(train_df[features], test_df[features], how='inner')

    print("=" * 69)
    print(f"There are {n_duplicates} duplicated rows in the training data set.")
    print(f"There are {overlap.shape[0]} overlapping observations that appear in both the train and test data sets")
    print("=" * 69)

    if verbose and n_duplicates > 0:
        print(train_df[duplicate_mask][features].head(10).T)
        print("=" * 69)
        plot_target_eda(train_df[duplicate_mask], target, title="target distribution by duplicated rows")
        print("=" * 69)
        #TODO: consider more visualizations comparing duplicated rows to non-duplicated rows in training data and to test data
        #TODO: show examples of duplicated rows, highlight differences in target values for duplicated rows, and visualize feature distributions for duplicated vs non-duplicated rows to look for patterns that could explain the duplicates or target differences
    if drop and n_duplicates > 0:
        print("Dropping duplicated rows from training data set...")
        df.loc[mask, :] = train_df.drop_duplicates(subset=features, keep="first")
        df = df.dropna(subset=features)
        train_df = df[mask] #reset train_df after dropping rows 
        duplicate_mask = train_df[features].duplicated(keep=False)
        overlap = pd.merge(train_df[features], test_df[features], how='inner')
        print(f"{duplicate_mask.sum()} duplicated rows remain in training data set.")
        print(f"{overlap.shape[0]} overlapping observations remain in the train and test data sets")
        print("=" * 69)
    return df

def get_features_with_na(df:pd.DataFrame, features:List[str], verbose:bool=True)->List[str]:
    """
    returns a list of features with null data in 
    """
    def _calculate_null(df):
        ds = (df[features].isnull().sum() / len(df)) * 100
        return ds

    #TODO combine plots into single plot
    def _plot_null(ds, title="Percentage of missing values in training data"):
        ds[:10].plot(kind = 'barh', title = f"Top {min(10, len(ds))} of {len(ds)} Features")
        plt.title(title)
        plt.xlabel('Percentage')
        plt.show()

    def _plot_missing_heatmap(df, title="null value heatmap"):
        cmap = _get_cmap()
        sample_size = min(df.shape[0], 1000)
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.sample(sample_size).isnull(), cbar=False, cmap=cmap, ax=axs[1])
        plt.title(title)
        plt.show()

    def _plot_missing_data(df, ds):
        fig, axs = plt.subplots(nrows=1, ncols=2)
        #bar plot
        ds.sort_values(ascending=False, inplace = True)
        ds = ds[ds > 0]
        axs[0] = ds[:10].plot(kind = 'barh', title = f"Top {min(10, len(ds))} of {len(ds)} Features")
        #heat map
        cmap = _get_cmap()
        sample_size = min(df.shape[0], 1000)
        sns.heatmap(df.sample(sample_size).isnull(), cbar=False, cmap=cmap, ax=axs[1])
        plt.show()


    if 'target_mask' in df.columns:
        mask = df.target_mask.eq(True)
        ds_train = _calculate_null(df[df.target_mask.eq(True)])
        ds_test = _calculate_null(df[df.target_mask.eq(False)])
    else:
        print("Unable to separate test and training data")
        ds = _calculate_null(df)
        ds = ds[ds > 0]
        print("Features with missing data:")
        print(ds)
        return ds.index.tolist()

    if (ds_train == 0).all() and (ds_test == 0).all():
        print("=" * 69)
        print("No Missing Data")
        print("=" * 69)
        return []
    else:
        print("=" * 69)
        print("                    Percent Null")
        print("=" * 69)
        df_plot = ds_train.to_frame(name = 'training data')
        df_plot['test data']  = ds_test
        df_plot = df_plot.loc[~(df_plot == 0).all(axis=1)]
        df_plot.round(4)
        print(df_plot)
        if verbose:
            _plot_null(ds_train)
            _plot_missing_heatmap(df)
        return df_plot.index.tolist()

def clean_strings(df: pd.DataFrame, features: List[str], 
                       string_length: int=5, fillna:bool=True) -> pd.DataFrame:
    """
    edits strings in feature columns to be more consistent and easier to work with
    -----------
    updated df with cleaned strings assigned as categorical features
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
            .str.replace(". ", "_", regex=False)
            .str.replace(", ", "_", regex=False)
            .str.replace(" ", "_", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[col] = df[col].str[:string_length].astype('category')
    return df

def _get_mean_std(df, feature, target):
    if "target_mask" in df.columns:
        df_group = df[df.target_mask.eq(True)]
    else: 
        df_group = df.copy()
    
    std = df_group[target].std()
    group_stats = df_group.groupby(feature)[target].agg(['mean', 'std']).rename(
            columns={
                'mean': f'{target}_by_{feature}_mu',
                'std': f'{target}_by_{feature}_std'}) 
    df = df.merge(group_stats, on=feature, how='left')
    df[f'{target}_by_{feature}_std'] = df[f'{target}_by_{feature}_std'].fillna(2*std)
    return df

def denoise_categoricals(df: pd.DataFrame, features: List[str], 
                         target: Optional[str]=None, threshold:float=0.05)-> pd.DataFrame:
    """
    identifies and consolidates noise values in categorical features
    ---------
    returns: df with consolidated noise in categorical features 
    optionally adds new features of target mean and std if target is provided
    ---------
    noise is defined as values that appear in 
       - only train data
       - only test data
       - appear infrequently (below threshold) 
            for 0.05% threshold this is 5 in 10000 or 500 in 1 million samples
            raising threshold will identify more noise values
    """
    if 'target_mask' in df.columns:
        df_train = df[df.target_mask.eq(True)]
        df_test = df[df.target_mask.eq(False)]
    else:
        print("Unable to separate test and training data")
        return df

    # assume threshold is given in percent
    # this is the minimum number of values required to be considered signal vs noise
    noise_ceil_train = int(0.01 * threshold * df_train.shape[0])
    noise_ceil_test = int(0.01 * threshold * df_test.shape[0])

    cat_features = [f for f in features if (df[f].dtype=='category' or df[f].dtype=='object' or df[f].dtype=='int')]
    other_features = [f for f in features if f not in cat_features]
    if other_features is not None:
        print(f"Features not evaluated for noise. Check data_type: {other_features}")

    for feature in cat_features:
        df[feature] = df[feature].astype('category')
        noise_label = -1 if (df[feature].cat.categories.dtype == 'int' or 
                             df[feature].cat.categories.dtype == 'float') else "noise"
        train_v = list(df_train[feature].unique())
#        test_v = list(df_test[feature].unique())
#        train_noise = [f for f in train_v if f not in test_v]
#        test_noise = [f for f in test_v if f not in train_v]
#        values = train_v + test_noise
        values = list(df[feature].dropna().unique())
        if len(train_v) < 2:
            print(f"{feature} is trivial, dropping {feature}")
            df.drop(columns=[feature], inplace=True)
        else:
            # Identify noise
            noise_dict = {}
            # noise includes all values that uniquely appear in train or test data
#            for v in test_noise + train_noise:
#                noise_dict[v] = noise
            # noise includes all values that appear infrequently in train or test data
            for v in values:
                if (df_train.groupby(feature)[feature].count()[v] < noise_ceil_train or
                    df_test.groupby(feature)[feature].count()[v] < noise_ceil_test):
                    noise_dict[v] = noise
            if len(noise_dict.keys()) == 0:
                print(f"No noise identified in {feature}")
                if target is not None:
                    df = _get_mean_std(df, feature, target)
            elif len(noise_dict.keys()) == 1:
                print(f"❌ Unable to denoise {feature}. Only one noise value: {noise_dict.keys()}.")
            else:
                df_train[f"{feature}_denoise"] = df_train[feature].replace(noise_dict)
                training_noise = df_train[df_train[f"{feature}_denoise"].eq(noise)].shape[0]
                if  training_noise > 0:
                    df[feature] = df[feature].replace(noise_dict)
                    df[feature].fillna(noise_label, inplace=True)
                    print(f"✔️ successfully de-noised {feature}: {100*training_noise/df_train.shape[0]:.2f}% noise in training data")
                    if target is not None:
                        df = _get_mean_std(df, feature, target)
                else:
                    print(f"""❌ Unable to denoise {feature}.\n
                              training noise: {training_noise} samples with noise in test data\n
                              noise values in test data: {list(noise_dict.keys())}
                          """)
    return df 

def get_outliers(df: pd.DataFrame, feature: str, deviations: int=4, 
                 remove: bool=False, verbose: bool=False) -> pd.DataFrame:
    """
    identifies outliers in a specified feature of a DataFrame
    -----------
    returns: DataFrame with outlier rows optionally removed, and DataFrame of outliers
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
    df[f'outlier_{name}'] = df[f'outlier_{name}'].astype('category')
    outlier_count = df.query(f'outlier_{name} == -1').shape[0]
    print("=" * 69)
    print(f"\n{outlier_count} ({100 * outlier_count / df.shape[0] :.2f}%) of samples identified as outliers")
    if outlier_count == 0:
        df.drop(f'outlier_{name}', inplace=True, axis=1)
    if drop is True:
        #TODO fix to only drop outliers from training data - can't drop test data!!!
        outlier_mask = df[df[f'outlier_{name}'] == -1]
        df.drop(outlier_mask.index, inplace=True)
        df.drop(f'outlier_{name}', inplace=True, axis=1)
        print(f"Removed {outlier_mask.shape[0]} outliers from original DataFrame.")
    return df

########################################
# Feature Engineering

def _plot_embeddings(df: pd.DataFrame, x:str, y:str, target:Optional[str]=None) -> None:
    """ helper function for clean scatterplots """
    if target != None and df[target].nunique() > 10:
        palette = _get_cmap()
    else:
        palette = _get_colors()

    df_plot = df.sample(n=min(750, df.shape[0]), random_state=69)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.scatterplot(data=df_plot, x=df_plot[x], y=df_plot[y], 
                    hue=target, alpha = 0.7, ax=ax, palette=palette, legend=False)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("")
    plt.ylabel("")
    plt.show()
    return

def get_embeddings(df: pd.DataFrame, features:List[str], mapper, col_names:str, 
    sample_size: Optional[float]=None, target:Optional[str]=None, encode_dim:int=8, index: int=1, verbose:bool=True) -> pd.DataFrame:
    """
    fits a mapper to a sample of the data, applys the mapping to the full dataset to create new features
    useful for generating embeddings or kernal approximations of selected feature space
    mapper can be "UMAP", "PCA", "PLS" or function with fit/transform methods
    -----------
    returns: df with new features added for the encoding
    """
    if 'target_mask' in df.columns:
        df_train = df[df.target_mask.eq(True)]
    else:
        df_train = df.copy()

    if sample_size is not None:
        if sample_size < 1.0:
            n = min(int(df_train.shape[0] * sample_size), 25000)
        else:
            n = min(int(df_train.shape[0]), int(sample_size))
        df_train = df_train.sample(n=n, random_state=69)

    if mapper == "UMAP":
        mapper = umap.UMAP(n_neighbors=8, low_memory=True, n_components=encode_dim)
    elif mapper == "PCA":
        mapper = skl.decomposition.PCA(n_components=encode_dim)
    elif mapper == "PLS":
        mapper = skl.cross_decomposition.PLSRegression(n_components=encode_dim)

    print(f"Fitting embedding function on {len(df_train)} samples....")
    tic=time()
    mapper = mapper.fit(df_train[features])

    print("Transforming features to embeddings...")
    reduced_data = mapper.transform(df[features])
    
    cols = [(col_names + str(i)) for i in range(index, index + reduced_data.shape[1])]
    new_features = pd.DataFrame(reduced_data, columns=cols, index=df.index)
    
    print(f"Added {len(cols)} {col_names} embedding features in {time()-tic:.2f}sec")
    df = df.join(new_features)
    if verbose:
        _plot_embeddings(df, cols[0], cols[1], target)
    return df

def get_target_transformer(df: pd.DataFrame, target: str, 
                           targets: list, name: str="enc",
                           TargetTransformer=None, 
                           get_dummies: bool=False,
                           verbose: bool=True
                           ):
    """
    scales and/or transforms targets in df with scikit learn scalers / transformers
    defaults to StandardScaler for numeric data and label encoder for categorical data
    -----------
    returns tuple with:
    - df with transformed target
    - updated list of targets with new transformed target column name added
    - fitted TargetTransformer to support inverse transformation of predictions
    -----------
    """
    if TargetTransformer == None:
        if df[target].dtype == float or df[target].dtype == int:
            TargetTransformer=skl.preprocessing.StandardScaler()
        else:
            TargetTransformer=skl.preprocessing.LabelEncoder()

    t=target.casefold().strip().replace(" ","_").replace("(","_").replace(")","").replace("-","_").replace(".","_")
    #initialize new features with -1
    enc_tgt = f"{t}_{name}"
    df[enc_tgt] = -1
    #only fit and transform valid target fields
    mask = df.get("target_mask", pd.Series(True, index=df.index))
    y_fit = df.loc[mask, target].values.reshape(-1, 1)
    y_transformed = TargetTransformer.fit_transform(y_fit).ravel()
    df.loc[mask, enc_tgt] = y_transformed

    if verbose:
        print(f"Added transformed target '{t}_{name}' to DataFrame")
        plot_target_eda(df, enc_tgt, title=f"Distribution of Transformed Target: {enc_tgt}")

    if (df[target].dtype == "O" or df[target].dtype == "category") and get_dummies==True:
        # TODO: Need to test this - perhaps better to drop the "t-1" features?
        df[enc_tgt] = df[enc_tgt].astype('category')
        df_dummies = pd.get_dummies(df.loc[mask, enc_tgt], dtype=int, prefix=t)
        new_cols = df_dummies.columns.tolist()
        df = df.join(df_dummies)
        df[new_cols].fillna(-1, inplace=True)
        targets = targets + new_cols
        if verbose: print(f"Added {len(new_cols)} binary classification targets by one hot encoding")

    targets = targets + [enc_tgt]
    return df, targets, TargetTransformer

def get_transformed_features(df: pd.DataFrame, features: List[str], FeatureTransformer, winsorize: tuple=[0,0]):
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

def get_feature_interactions(df: pd.DataFrame, features:List[str], winsorize: tuple=[0,0], 
                             self_transform: bool=False,
                             transform: bool=True) -> pd.DataFrame:
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
    if self_transform:
        for feature in features:
            df[f'{feature}_sq'] = df[feature] ** 2
            df[f'{feature}_cube'] = df[feature] ** 3
            if np.min(df[feature]) >= 0:
                df[f'{feature}_log'] = np.log1p(df[feature])  
                df[f'{feature}_sqrt'] = np.sqrt(df[feature])
        if transform:
            new_features = [f for f in df.columns if ('_log' in f or '_sqrt' in f or '_cube' in f or '_sq' in f)]
            df = get_transformed_features(df, new_features, skl.preprocessing.MinMaxScaler(), winsorize=winsorize)

    for combination in tqdm(itertools.combinations(features, 2), desc="Creating interaction features", unit="pairs"):
        df["*".join(combination)] = df[list(combination)].prod(axis=1)
    new_features = ["*".join(c) for c in itertools.combinations(features, 2)]
    if transform:
        df = get_transformed_features(df, new_features, skl.preprocessing.PowerTransformer(), winsorize=winsorize)
    print(f"Added {len(new_features)} inteaction features")
    return df

def get_feature_by_grouping_on_cat(df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
    """
    Computes mean and std of a numeric target feature grouped by each category,
    using only rows where df.target_mask == True, and merges the results back.
    """
    for feature in features: 
        _get_mean_std(df, feature, target)
    return df

def get_feature_cat_interactions(df: pd.DataFrame, features:list, pivot:str)-> pd.DataFrame:
    """
    returns each combination of category values as a new category
    --------
    remember to check categories after
    --------
    requires: pandas, scikit learn, numpy
    """
    for feature in features:
        df[f'{feature}_on_{pivot}'] = df[feature].astype(str) + df[pivot].astype(str)
        if df[f'{feature}_on_{pivot}'].nunique() < 256:
            X = df[f'{feature}_on_{pivot}'].values
            df[f'{feature}_on_{pivot}'] = skl.preprocessing.OrdinalEncoder().fit_transform(X.reshape(-1, 1))
            df[f'{feature}_on_{pivot}'] = df[f'{feature}_on_{pivot}'].astype('int').astype('category')
        else:
            print(f"{feature}_on_{pivot} has {df[f'{feature}_on_{pivot}'].nunique()} values and needs encoding")
    return df

def get_target_hints(df:pd.DataFrame, features:list, target:str, model=None, folds=7, index_id: str="1", task: str='regression') -> pd.DataFrame:
    """
    adds new features using by generating “hint” features via cross‑validated out‑of‑fold predictions and ensembles.
    -------
    returns:
    df: with updated features
    --------
    requires:
    pandas, numpy, scikit learn 
    """
    mask = df.get("target_mask", pd.Series(True, index=df.index))
    X = df.loc[mask, features]
    X_test = df.loc[~mask, features]
    y = df.loc[mask, target]

    if task.startswith('regression'):
        cv = skl.model_selection.KFold(n_splits=folds, shuffle=True, random_state=80085)
        base_model = skl.linear_model.Ridge() if model is None else model
        y_hint = np.zeros(len(df))
    else:
        cats = y.unique()
        cv = skl.model_selection.StratifiedKFold(n_splits=folds, shuffle=True, random_state=80085)
        base_model = skl.naive_bayes.GaussianNB() if model is None else model
        y_hint = np.zeros((len(df), len(cats)))

    models = []
    for (train_idx, val_idx) in tqdm(cv.split(X, y), total=cv.get_n_splits()):
        m = base_model 
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        m.fit(X_train, y_train)
        if task.startswith('regression'):
            y_hint[X.index[val_idx]] = m.predict(X_val)
            y_hint[X_test.index] += m.predict(X_test) / 7
        else:
            y_hint[X.index[val_idx], :] = m.predict_proba(X_val)
            y_hint[X_test.index, :] += m.predict_proba(X_test) / 7
        models.append(m)

    if task.startswith('regression'):
        df[f"hint_{index_id}"] = y_hint
    else:
        cols = [f"hint_{index_id}_{str(cat)}" for cat in cats]
        df_hint = pd.DataFrame(y_hint, index=df.index, columns=cols)
        df = pd.concat([df, df_hint], axis=1)

    return df


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
    if target is not None and (df[target].dtype == 'float' or df[target].dtype == 'int'):
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
        
        if target is not None:
            plot_features_eda(df, [col_name], target)

        reduced_data = skl.decomposition.PCA(n_components=2).fit_transform(df[features])
        df[f'{col_name}_x'] = reduced_data[:,0]
        df[f'{col_name}_y'] = reduced_data[:,1]
        #TODO decide to keep or drop these PCA features
        _plot_embeddings(df, col_name+ '_x', col_name+ '_y', col_name)
        df.drop(columns=[col_name+ '_x', col_name+ '_y'], inplace=True)
        
    if not df[f"{col_name}_noise"].any():
        df.drop(columns=f"{col_name}_noise", inplace=True)
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
        _, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(df[cyclic_features[0]], df[cyclic_features[1]])
        ax.set_xlabel(None)
        ax.set_xticks([])
        ax.set_ylabel(None)
        ax.set_yticks([])
        plt.show()
        
    my_palette = _get_colors()
    
    if verbose:
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex = True, figsize=(8,3))
        sns.histplot(data=df[df.target_mask.eq(True)], x=feature, color = my_palette[0], 
                     ax=axs[0])
        sns.histplot(data=df[df.target_mask.eq(False)], x=feature, color = my_palette[2], 
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

def get_cycles_from_feature(df:pd.DataFrame, feature: str, points:float=None, verbose:bool=True)->pd.DataFrame:
    """
    decomposes a clock type feature into a sin/cos component
    use points = 7 for days of week 
    use points = 360 for compass headings
    ------
    returns df with updated features
    ------
    requires: pandas, numpy, seaborn
    """
    def _plot_circle(df, cyclic_features):
        _, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(df[cyclic_features[0]], df[cyclic_features[1]])
        ax.set_xlabel(None)
        ax.set_xticks([])
        ax.set_ylabel(None)
        ax.set_yticks([])
        plt.show()
    
    if points is None:
        points = df[feature].nunique()
    X = df[feature].astype(float).values.reshape(-1,1)
    df[f'{feature}_{points}_sin'] = np.sin(2 * np.pi * X/points)
    df[f'{feature}_{points}_cos'] = np.cos(2 * np.pi * X/points)
    
    if verbose:
        _plot_circle(df, [f'{feature}_{points}_sin', f'{feature}_{points}_cos'])

    return df

########################################
# Training
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

def calculate_score(actual, predicted, metric='rmse')-> float:
    """ 
    calculates score based on metric or task
    simplifies calling scikit learn metrics by allowing flexible metric names
    -----------
    returns: metric score
    for rmse use 'regression'
    for accuracy use 'classification'
    for roc_auc use 'probability'
    -----------
    requires: scikit learn
    """
    if 'rmse_log' in metric or 'rmsle' in metric or 'log_rmse' in metric:
        predicted = np.clip(predicted, 0, None)
        return skl.metrics.root_mean_squared_log_error(actual, predicted)
    elif 'rmse' in metric or metric == 'regression':
        return skl.metrics.root_mean_squared_error(actual, predicted)
    elif 'accuracy' in metric or metric=='classification':
        return skl.metrics.accuracy_score(actual, predicted)
    elif 'auc' in metric or metric == 'probability':
        return skl.metrics.roc_auc_score(actual, predicted)
    elif 'r2' in metric:
        return skl.metrics.r2_score(actual, predicted)
    elif 'mae' in metric:
        return skl.metrics.mean_absolute_error(actual, predicted)
    elif 'mape' in metric:
        return skl.metrics.mean_absolute_percentage_error(actual, predicted)
    elif 'f1' in metric:
        return skl.metrics.f1_score(actual, predicted)
    elif 'precision' in metric:
        return skl.metrics.precision_score(actual, predicted)
    elif 'log_loss' in metric:
        return skl.metrics.log_loss(actual, predicted)
    elif 'brier' in metric:
        return skl.metrics.brier_score_loss(actual, predicted)
    elif 'weighted_top' in metric:
        top_k = 3 #TODO update to pull digit in metric following substring "_top_", default to 3 if not provided
        score_j = 0
        score=0
        for i in range(top_k):
            score_i = skl.metrics.top_k_accuracy_score(actual, predicted, k = i+1)
            score += (score_i - score_j) / (i+1)
            score_j = score_i
        return score
    elif '_top' in metric:
        top_k = 3  #TODO update to pull digit in metric following substring "_top_", default to 3 if not provided
        return skl.metrics.top_k_accuracy_score(actual, predicted, k = top_k)
    else:
        raise ValueError("""***UNSUPPORTED METRIC***\n
            Supported regression metrics: 'rmse', 'mae', 'r2', 'mape', 'rmsele' \n
            Supported classification and probability metrics: 'accuracy', 'precision', 'roc_auc', 'f1', 'log_loss', 'brier', 'top_k'""")

def plot_training_results(X_t, X_v, y_t, y_v, y_p, task: str='regression', embed_v=None)-> None:
    """
    plots training results with reference model for comparison
    -----------
    fits a base model to the training data (X_t, y_t) to provide a reference point for evaluating the trained model predictions
    scores the base model predictions and trained model predictions (y_p) against validation data (y_v) 
    based on the task (regression, classification, or probability)
    plots:
    for regression: actual vs predicted scatterplots for trained model and ridge regression, distribution of predictions vs actuals, and residual distribution
    for classification: confusion matrices for trained model and gaussian naive bayes, distribution of predicted probabilities; good for binary classification and multiclassification
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
    
    if task.startswith("probability"):
        if y_p.shape != y_v.shape:
            y_p = np.argsort(y_p, axis=1)[:, -1:][:, ::-1]
            task="classification"
            y_base = base_model.predict(X_v[numeric_features]).reshape(-1, 1)
        else:
            y_base = base_model.predict_proba(X_v[numeric_features])[:, 1].reshape(-1, 1)
    else:
        y_base = base_model.predict(X_v[numeric_features]).reshape(-1, 1)
    
    def _plot_regression_resid(ax):
        skl.metrics.PredictionErrorDisplay.from_predictions(y_v[:1000], y_base[:1000], kind = 'actual_vs_predicted',
                                                            scatter_kwargs={"color":'xkcd:gold', "alpha":0.8},
                                                            ax = ax)
        skl.metrics.PredictionErrorDisplay.from_predictions(y_v[:1000], y_p[:1000], kind = 'actual_vs_predicted', 
                                                            scatter_kwargs={"alpha":0.8},
                                                            line_kwargs={"color":'xkcd:dusty rose'},
                                                            ax = ax)
        metric = 'rmse' if task=="regression" else task.split("_")[1]
        ax.set_title(f"Trained Model {calculate_score(y_v, y_p, metric = metric):.4f} vs Ridge {calculate_score(y_v, y_base, metric = metric):.4f} {metric.upper()}")

    def _plot_classification_cm(ax, predictions=y_p, title = "Trained", show_values=True):
        cmap='bone_r' if len(np.unique(predictions)) < 4 else 'cividis'
        skl.metrics.ConfusionMatrixDisplay.from_predictions(y_v, predictions, cmap=cmap, include_values=show_values,
                                                            normalize='all', colorbar=False, ax=ax)
        ax.invert_yaxis()
        ax.set_title(f"{title} Model Accuracy {100*calculate_score(y_v, predictions, metric = 'accuracy'):.1f}%")

    def _plot_classification_roc(ax):
        skl.metrics.RocCurveDisplay.from_predictions(y_v, y_p, ax=ax, name="Trained Model")
        skl.metrics.RocCurveDisplay.from_predictions(y_v, y_base, name="GaussianNB", ax=ax)
        ax.set_title("ROC Curve")

    def _plot_distribution(ax, hist=True):
        if hist == True:
            ax.hist(y_v, bins=min(50,len(np.unique(y_v))), color='xkcd:silver', alpha=0.8, density = True)
            ax.hist(y_p, bins=min(50,len(np.unique(y_v))), color='xkcd:ocean blue', alpha=0.9, density = True)
        else: 
            order = sorted(np.unique(y_v))
            s = pd.Series(y_v.ravel())
            counts = s.value_counts().reindex(order, fill_value=0)
            ax.bar(counts.index, counts.values, width=0.9, align='center', color='xkcd:silver', alpha=0.8)
            s = pd.Series(y_p.ravel())
            counts = s.value_counts().reindex(order, fill_value=0)
            ax.bar(counts.index, counts.values, width=0.85, align='center', color='xkcd:ocean blue', alpha=0.9)
        ax.set_ylabel("Probability Density")
        ax.set_title("Prediction vs Training Distribution")
        ax.set_yticks([])

    def _plot_residuals(ax):
        residuals = y_p - y_v
        ax.hist(residuals, bins=min(50,2+len(np.unique(residuals))), color='xkcd:dull green', alpha=0.9)
        ax.set_title("Residual Distribution")
        ax.set_yticks([])
        ax.set_ylabel("Count")

    def _plot_embeddings(ax, proj, color_by, title, cmap='cividis'):
        ax.scatter(proj[:, 0], proj[:, 1], c=color_by, cmap=cmap, s=5, alpha=0.7)
        ax.set_title(title)
        ax.axis('off')

    col = 3 if embed_v is None else 4

    fig = plt.figure(figsize=(3*col, 6))
    gs = mpl.gridspec.GridSpec(2, col, figure=fig)

    if task.startswith("regression"):
        _plot_regression_resid(fig.add_subplot(gs[:, :2]))
        _plot_distribution(fig.add_subplot(gs[0, 2]))
        _plot_residuals(fig.add_subplot(gs[1, 2]))

    elif task.startswith("classification"):
        _plot_classification_cm(fig.add_subplot(gs[:, :2]))
        _plot_distribution(fig.add_subplot(gs[0, 2]), hist=False)
        _plot_classification_cm(fig.add_subplot(gs[1, 2]), 
                               predictions = y_base, 
                               show_values=True if len(np.unique(y_v)) < 4 else False, 
                               title = "GaussianNB")
    
    elif task.startswith("probability"):
        _plot_classification_roc(fig.add_subplot(gs[:, :2]))
        _plot_distribution(fig.add_subplot(gs[0, 2]))
        _plot_classification_cm(fig.add_subplot(gs[1, 2]), predictions=np.round(y_p))

    if embed_v is not None:
        pca_projection = skl.decomposition.PCA(n_components=2)
        pca_proj = pca_projection.fit_transform(embed_v)
        residuals = y_p - y_v
        residuals = np.clip(residuals, np.percentile(residuals, 1), np.percentile(residuals, 99)) 
        _plot_embeddings(fig.add_subplot(gs[0, 3]), pca_proj, y_v, "Embedding - Target")
        _plot_embeddings(fig.add_subplot(gs[1, 3]), pca_proj, residuals, "Embedding - Residual", cmap='bwr')

    plt.tight_layout()
    plt.show()

########################################
### Train and evaluate
def train_and_score_model(X_train: pd.DataFrame, X_val:pd.DataFrame, 
                          y_train: pd.Series, y_val:pd.Series,
                          model, task: str="regression", 
                          verbose: bool=True, 
                          TargetTransformer=None):
    """
    trains a model and returns trained model & score
    -----------
    model: a scikit learn compatible model with fit and predict (and predict_proba) methods
    task: "regression", "classification", or "probability" to determine prediction and scoring method
    returns:
    - trained model
    - score based on task
    -----------
    requires: pandas, scikit learn, numpy, matplotlib
    """
    model.fit(X_train, y_train)
    if task.startswith("regression") or task.startswith("classification"): y_predict = model.predict(X_val)
    elif task.startswith("probability"): 
            n_cats = y_train.nunique()
            if n_cats == 2:
                y_predict = model.predict_proba(X_val)[:, 1]
            else:
                y_predict = model.predict_proba(X_val)
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
        y_p = y_predict.reshape(y_v.shape[0], -1)
    score = calculate_score(y_v, y_p, metric = task)
    print(f"***  model score:  {score:.4f}  ***")
    if verbose == True: 
        plot_training_results(X_train, X_val, y_t, y_v, y_p,
                              task=task) 
    return model, score

def train_and_score_multiclass_model(X_train: pd.DataFrame, X_val:pd.DataFrame, 
                                    y_train: pd.Series, y_val:pd.Series,
                                    model, raw_score: bool=False,  top_k: int=3,
                                    verbose: bool=True):
    """
    trains a multiclass prediction model and returns trained model & top k score
    -----------
    model: a scikit learn compatible model with fit and predict methods
    -----------
    returns:
    - trained model
    - top k score 
    -----------
    requires: pandas, scikit learn, numpy, matplotlib
    """
    model.fit(X_train, y_train)
    y_predict = model.predict_proba(X_val)
     # unweighted top_k accuracy
    if raw_score:
        score = skl.metrics.top_k_accuracy_score(y_val, y_predict, k = top_k)
    # weighted top_k accuracy
    else:
        score_j = 0
        score=0
        for i in range(1,top_k+1):
            score_i = skl.metrics.top_k_accuracy_score(y_val, y_predict, k = i)
            score += (score_i - score_j) / i
            score_j = score_i

    print(f"***  model {'raw' if raw_score==True else 'weighted'} top-{top_k} score:  {score:.4f}  ***")
    if verbose == True: 
        y_t = np.array(y_train).reshape(-1, 1)
        y_v = np.array(y_val).reshape(-1, 1)
        y_predict = model.predict(X_val)
        y_p = np.array(y_predict).reshape(-1, 1)
        #TODO: evaluate better training plots for multiclass performance
        plot_training_results(X_train, X_val, y_t, y_v, y_p,
                              task="classification") 
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
                                     metric: str='classification', direction: str=None,
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
    if direction is None:
        if metric.startswith("regression"):
            direction='minimize'
        else:
            direction='maximize'

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
            if metric.startswith("probability"): 
                study_params['metric'] = 'auc'
                model = lgb.LGBMClassifier(**study_params, random_state=SEED)
            elif metric.startswith("classification"):
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

def submit_predictions(X: pd.DataFrame, y: pd.Series, target: str, 
                       models: list, task: str='regression', TargetTransformer=None, 
                       path: str="", file: str="sample_submission.csv", verbose: bool=True, )-> pd.DataFrame:
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

    submission_df = pd.read_csv(f"{path}/{file}")
    submission_df[target] = y_pred
    submission_df.to_csv(f'/kaggle/working/submission.csv', index=False)

    if verbose:
        plot_target_eda(submission_df, target, title = f"distribution of {target} predictions")

    print("=" * 6, 'save success', "=" * 6, "\n")
    print(f"Predicted target mean: {y_pred.mean():.4f} +/- {y_pred.std():.4f}")
    return submission_df

def cv_train_models(df: pd.DataFrame, features: dict, target: str, models: dict,
                    task: str = "regression", folds: int = 7, meta_model=None,
                    TargetTransformer=None, verbose: bool = True, more_oof=None):
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
    n_cats = y.nunique() if task.startswith("probability") else 1
    
    trained_models = {}
    model_names = list(models.keys())
    n_models = len(model_names)
    extra_cols = 0
    if more_oof is not None:
        more_oof = more_oof.reshape(y.shape[0], -1)
        extra_cols = more_oof.shape[1]

    if n_cats > 2: 
        oof_matrix = np.zeros((y.shape[0], n_cats, n_models + extra_cols))
        if extra_cols > 0:
            oof_matrix[:, :, n_models:] = more_oof
    else:
        oof_matrix = np.zeros((y.shape[0], n_models + extra_cols))
        if extra_cols > 0:
            oof_matrix[:, n_models:] = more_oof
    
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
        if n_cats > 2: 
            oof_pred = np.zeros((y.shape[0], n_cats))
        else:
            oof_pred = np.zeros((y.shape[0], 1))

        for (train_idx, val_idx) in tqdm(cv.split(X, y), desc="training models", unit="folds"):
            X_t, X_v = X[features[k]].iloc[train_idx], X[features[k]].iloc[val_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]

            try:
                model.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            except Exception:
                model.fit(X_t, y_t)

            if task.startswith("probability"):
                y_v_pred = model.predict_proba(X_v)
            else:
                y_v_pred = model.predict(X_v)

            if n_cats > 2: 
                oof_pred[val_idx, :] = y_v_pred
            elif n_cats ==2: 
                oof_pred[val_idx] = y_v_pred[:, 1].reshape(-1, 1)
            else:
                oof_pred[val_idx] = y_v_pred.reshape(-1,1)

            cv_models.append(model)

        trained_models[k] = cv_models

        if n_cats > 2:
            oof_matrix[:, :, i] = oof_pred
        else:
            oof_matrix[:, i] = oof_pred.ravel()

        if TargetTransformer is not None:
            oof_pred = TargetTransformer.inverse_transform(oof_pred)
            y_all = TargetTransformer.inverse_transform(
                np.array(y).reshape(-1,1))
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
        else:
            y_all = np.array(y).reshape(-1, 1)
        score = calculate_score(y_all, oof_pred, metric=task)
        print(f"***  model {k} final cv score:  {score:.4f}  ***")

        if verbose:
            # note: plotting last fold's X_t, X_v, y_t, y_v, and corresponding preds
            if n_cats == 2:
                plot_training_results(X_t, X_v, y_t, y_v, y_v_pred[:, 1],
                    task=task)
            else:
                plot_training_results(X_t, X_v, y_t, y_v, y_v_pred,
                    task=task)

    # ---- Stacking meta-model on OOF predictions ----
    if meta_model == None:
        if task.startswith("regression"):
            meta_model = skl.linear_model.Ridge(alpha=1.0)
        else:
            meta_model = skl.linear_model.LogisticRegression()
    if n_cats > 2:
        oof_matrix = oof_matrix.reshape(y.shape[0], -1)

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
                      TargetTransformer=None, meta_model=None, more_oof=None,
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
        extra_cols = 0
        cols = len(models.keys())
        if more_oof is not None:
            more_oof = more_oof.reshape(y.shape[0], -1)
            extra_cols = more_oof.shape[1]
        y_oof_matrix = np.zeros((y.shape[0], cols + extra_cols))
        if extra_cols > 0:
            y_oof_matrix[:, cols:] = more_oof

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

def cv_train_multiclass_models(df: pd.DataFrame, features: dict, target: str, models: dict,
                                folds: int = 7, meta_model=None, task: str="classification_weighted_top_3",  top_k: int=3,
                                verbose: bool = True):
    
    """
    trains models with cross validation and returns trained models and a stacking meta model
    -----------
    for each model in models, trains with cross validation using the corresponding feature subset in features
    returns a dictionary of trained models and a meta stacking model
    prints OOF validation score for each model
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
    n_cats = y.nunique()

#    OOF matrix only needed for training meta model
    oof_matrix = np.zeros((y.shape[0], n_cats, n_models))
    
    for i, (k, model) in enumerate(models.items()):
        print(f"Training Model: {k}")
        cv = skl.model_selection.StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=69 + i
            )

        cv_models = []
        oof_pred = np.zeros((y.shape[0], n_cats))
        for (train_idx, val_idx) in tqdm(cv.split(X, y), desc="training models", unit="folds"):
            X_t, X_v = X[features[k]].iloc[train_idx], X[features[k]].iloc[val_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_t, y_t)
            y_v_pred = model.predict_proba(X_v)
            oof_pred[val_idx, :] = y_v_pred
            cv_models.append(model)

        trained_models[k] = cv_models
        oof_matrix[:, :, i] = oof_pred
        y_all = np.array(y).reshape(-1, 1)

        score = calculate_score(y_all, oof_pred, metric=task)
        print(f"***  model {'raw' if raw_score==True else 'weighted'} top-{top_k}  cv score:  {score:.4f}  ***")

    # ---- Stacking meta-model on OOF predictions ----
    tic=time()
    meta_model = skl.linear_model.LogisticRegression() if meta_model is None else meta_model
    print(f"Selected meta model is: {meta_model}")
    oof_matrix = oof_matrix.reshape(y.shape[0], -1)
    print(f"Training Meta Model on {oof_matrix.shape[1]} OOF predictions and {y.shape[0]} samples...")
    meta_model.fit(oof_matrix, y)
    print(f"Meta Model training completed in {time()-tic:.2f}sec")

    return trained_models, meta_model

def submit_cv_multiclass_predict(X: pd.DataFrame, y: pd.DataFrame, features: dict, target:str, 
                      models: dict, TargetTransformer=None, meta_model=None, top_k: int=3,
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
    def _plot_target(df: pd.DataFrame, picks: str, title: str='predicted target distribution') -> None:
        df_long = df[picks].melt(var_name="seq", value_name="value")
        sns.countplot(data=df_long, x="value", hue="seq", stat='percent', dodge=True)
        plt.yticks([])
        plt.xlabel("")
        plt.title(title)
        plt.show()

    n_models = len(list(models.keys()))
    n_cats = len(models[list(models.keys())[0]][0].classes_)
    
    if meta_model is None:
        y_test = np.zeros((y.shape[0], n_cats))
    else:
        y_oof_matrix = np.zeros((y.shape[0], n_cats, n_models))

    for i, (k, cv_models) in tqdm(enumerate(models.items()), desc="Training models", unit="models"):
        y_cv = np.zeros((y.shape[0], n_cats))
        training_features = features[k]
        for model in tqdm(cv_models, desc=f"Cross validated {k} models", unit="models"):
            y_cv += model.predict_proba(X[training_features])
            
        y_cv /= len(cv_models)
        if meta_model is None:
            y_test += y_cv
        else:
            y_oof_matrix[:, :, i] = y_cv

    if meta_model is None:
        y_test /= len(models.keys())
    else:
        y_oof_matrix = y_oof_matrix.reshape(y.shape[0], -1)
        y_test = meta_model.predict_proba(y_oof_matrix)

    #top_k indices (column #s in y_predict)
    topk_indices = np.argsort(y_test, axis=1)[:, -top_k:][:, ::-1]
   
    if TargetTransformer is not None:
        topk_cats = TargetTransformer.inverse_transform(topk_indices.flatten()).reshape(-1, top_k)
    else:
        topk_cats = topk_indices

    results = {}
    for i in range(top_k):
        results[f'pick_{i+1}'] = topk_cats[:, i]
    results['top3_combined'] = [f"{cat1} {cat2} {cat3}" 
                          for cat1, cat2, cat3 in zip(topk_cats[:, 0], topk_cats[:, 1], topk_cats[:, 2])]

    df_results = pd.DataFrame(results)    
    submission_df = pd.read_csv(f"{path}/{file}")
    submission_df[target] = results['top3_combined']
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)

    if verbose:
        _plot_target(df_results, list(results.keys()), title = f"distribution of {target} predictions")
    print("=" * 6, 'save success', "=" * 6, "\n")

    return submission_df

########################################
# Neural Net -> build a nn feature extractor network
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.ReLU()
    def forward(self, X):
        return self.activation(X + self.block(X))

class NeuralNetModel(nn.Module):
    def __init__(self, input_dim, task="regression",
                 embed_dim=256, num_classes=None, 
                 dropout_rate=0.1, noise_scale=0.002):

        super().__init__()
        self.embed_dim = embed_dim
        self.task = task
        self.noise_scale = noise_scale
        self.num_classes = num_classes
        
        if "ordinal" in task:
            self.output_dim = num_classes - 1
        elif task.startswith("classification") or task.startswith("probability"):
            self.output_dim = 1 if num_classes <=2 else num_classes
        else:
            self.output_dim = 1
        
        if "ordinal" in task and self.output_dim <= 1:
            raise ValueError("Ordinal task requires output_dim >= 2 (i.e., at least 3 classes).")

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2*embed_dim),
            nn.BatchNorm1d(2*embed_dim),
            nn.ReLU(),
            ResidualBlock(2*embed_dim),
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.GELU(),
            nn.Linear(2*embed_dim, 2*embed_dim),
            nn.GELU(),
            nn.Linear(2*embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(embed_dim // 2),
            nn.Linear(embed_dim // 2, 16),
        )
        # prediction head: raw logits (1D for regression or binary classification)
        self.predictor = nn.Sequential(
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(16, self.output_dim),
        )

    def __repr__(self):
        return f"NeuralNetModel(task={self.task}, embed_dim={self.embed_dim}, output_dim={self.output_dim})"

    def toggle_dropout(self, enable=True):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = m.p if enable else 0.0

    def encode(self, X, noise=True):
        embed = self.encoder(X.float())
        if noise:
            embed += torch.randn_like(embed) * self.noise_scale
        return embed

    @torch.no_grad()
    def get_embedding(self, X):
        return self.encode(X, noise=False)

    def forward(self, X):
        z = self.encode(X, noise=True)
        return self.predictor(z)

    def predict(self, X):
        z = self.encode(X, noise=False)
        return self.predictor(z)

    def predict_proba(self, X):
        if self.task.startswith("regression"):
            raise ValueError("predict_proba called on a regression task")

        logits = self.predictor(self.encode(X, noise=False))
        if "ordinal" in self.task:
            return logits
        elif self.output_dim == 1:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=1)

class RegressionLoss(nn.Module):
    def __init__(self, alpha=3.3):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))

    def cosine_weighted_mse(self, y_pred, y_true):
        with torch.no_grad():
            weight = 1.1 + torch.cos(math.pi * (y_true - self.alpha) / self.alpha)
        return torch.mean(weight * (y_pred - y_true) ** 2)

    def tophat_weighted_mse(self, y_pred, y_true):
        weight = (torch.abs(y_true) > 1).float()
        return torch.mean(weight * (y_pred - y_true) ** 2)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: raw model outputs BEFORE sigmoid/softmax
        targets: class indices (long) for multi-class, or float {0,1} for binary
        """
        if logits.size(-1) == 1:
            # Binary classification
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets.float(), reduction="none"
            )
            prob = torch.sigmoid(logits)
            pt = torch.where(targets == 1, prob, 1 - prob)
        else:
            # Multi-class classification
            ce = nn.functional.cross_entropy(
                logits, targets.long(), reduction="none"
            )
            prob = torch.softmax(logits, dim=1)
            pt = prob.gather(1, targets.unsqueeze(1)).squeeze()
            bce = ce

        focal = self.alpha * (1 - pt) ** self.gamma * bce

        return focal.mean() if self.reduction == "mean" else focal.sum()

class OrdinalLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.logits_dim = num_classes - 1  # K-1 binary tasks

    def forward(self, logits, targets):
        """
        logits: shape (N, K-1)
        targets: integer labels 0..K-1
        """
        # Convert targets to ordinal binary matrix
        ord_targets = (targets.unsqueeze(1) > torch.arange(self.logits_dim, device=targets.device)).float()
        return nn.functional.binary_cross_entropy_with_logits(logits, ord_targets)

def train_and_score_nn_model(
        X_train: pd.DataFrame, X_val: pd.DataFrame, y_train: pd.Series, y_val: pd.Series,
        model, DEVICE, task: str="regression", verbose: bool=True, TargetTransformer=None,
        num_epochs: int=100, lr: float=2e-5, patience_limit: int=5, batch_size: int=2048,
        save_path="/kaggle/working"):
    """
    """
    def _get_validation_predictions(logits_val, task=task):
        if task.startswith("regression"):
            preds_val = logits_val.numpy()
            if TargetTransformer != None:
                y_p = TargetTransformer.inverse_transform(preds_val.reshape(-1, 1))
            else:
                y_p = preds_val.reshape(-1, 1)
        elif "ordinal" in task:
            preds_val = (logits_val > 0).sum(dim=1).numpy()
            y_p = preds_val.reshape(-1, 1)
        #predict probability of classification
        elif task.startswith("probability"):
            if logits_val.ndim == 1:
                probs = torch.sigmoid(logits_val)
                y_p = probs.numpy().reshape(-1, 1)
            else:
                probs = torch.softmax(logits_val, dim=1)
                y_p = probs.numpy()  # shape (N, C)
        #predict classification labels
        else:
            if logits_val.ndim == 1:
                probs = torch.sigmoid(logits_val)
                preds_val = (probs > 0.5).long().numpy() #binary 1/0
            else:
                probs = torch.softmax(logits_val, dim=1)
                preds_val = probs.argmax(dim=1).numpy()  #multiclass labels
            y_p = np.array(preds_val).reshape(-1, 1)
        return y_p

    best_loss, best_model_state, patience = float('inf'), None, 0
    training_log = {}

    # DATA prep 
    X_train_tensor = torch.tensor(X_train.values.astype(np.float32)).to(DEVICE)
    X_val_tensor   = torch.tensor(X_val.values.astype(np.float32)).to(DEVICE)

    if task.startswith("regression"):
        y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).to(DEVICE)
        #y_val_tensor   = torch.tensor(y_val.values.astype(np.float32)).to(DEVICE)
    else:
        # classification → long labels
        y_train_tensor = torch.tensor(y_train.values.astype(np.int64)).to(DEVICE)
        #y_val_tensor   = torch.tensor(y_val.values.astype(np.int64)).to(DEVICE)

    if TargetTransformer != None:
        y_t = TargetTransformer.inverse_transform(y_train.values.reshape(-1, 1))
        y_v = TargetTransformer.inverse_transform(y_val.values.reshape(-1, 1))
    else:
        y_t = np.array(y_train).reshape(-1, 1)
        y_v = np.array(y_val).reshape(-1, 1)

    # Identify loss function and data shape 
    if task.startswith("regression"):
        num_classes = 1
        loss_fn = RegressionLoss()
    elif "ordinal" in task:
        num_classes = len(np.unique(y_train))
        loss_fn = OrdinalLoss(num_classes)
    elif task.startswith("classification") or task.startswith("probability"):
        num_classes = len(np.unique(y_train))
        num_classes = 1 if num_classes == 2 else num_classes 
        loss_fn = FocalLoss()   
    else:
        raise ValueError(f"Unknown task: {task}")

    # Learning parameters 
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # TRAINING LOOP
    for epoch in range(num_epochs):
        tick = time()
        model.train()
        loss_totals = 0

        perm = torch.randperm(X_train_tensor.size(0))
        X_train_tensor = X_train_tensor[perm]
        y_train_tensor = y_train_tensor[perm]

        num_batches = math.ceil(len(X_train_tensor) / batch_size)

        for batch in range(num_batches):
            optimizer.zero_grad()

            start = batch * batch_size
            end = min(start + batch_size, len(X_train_tensor))

            X_batch = X_train_tensor[start:end]
            y_batch = y_train_tensor[start:end]

            logits = model(X_batch)
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)

            loss = loss_fn(logits, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_totals += loss.item()

        scheduler.step()
        total_loss = loss_totals / num_batches

        # each epoch - validation
        model.eval()
        with torch.no_grad():
            logits_val = model.predict(X_val_tensor).cpu()

            if logits_val.shape[-1] == 1:
                logits_val = logits_val.squeeze(-1)

        y_p = _get_validation_predictions(logits_val)
        val_score = calculate_score(y_v, y_p, metric=task)

        # LOGGING
        training_log[epoch + 1] = [total_loss, val_score]

        if verbose and (epoch == 0 or (epoch + 1) % 5 == 0):
            print("=" * 69)
            print(f"Epoch {epoch+1} | Time: {time()-tick:.2f}s")
            print(f" Val Score: {val_score:.4f}, Pred μ: {y_p.mean():.3f}, σ: {y_p.std():.3f}")
        if verbose and (epoch == 0 or (epoch + 1) % 25 == 0):
            embeddings = model.get_embedding(X_val_tensor).detach().cpu().numpy()
            plot_training_results(X_train, X_val, y_t, y_v, y_p, task=task, embed_v=embeddings)

        # EARLY STOPPING
        if total_loss < best_loss:
            best_loss = total_loss
            best_model_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        if epoch > 10 and patience >= patience_limit:
            print(f"Early stopping triggered. Best Epoch {epoch + 1 - patience}.")
            break

    # FINALIZE
    training_log_df = pd.DataFrame.from_dict(
        training_log, orient='index',
        columns=['training_loss', 'val_score']
    )
    
    model.load_state_dict(best_model_state)
    if save_path is not None:
        torch.save(model.state_dict(), f"{save_path}/regression_model.pt")

    model.eval()
    with torch.no_grad():
        logits_val = model.predict(X_val_tensor).cpu()
        if logits_val.shape[-1] == 1:
            logits_val = logits_val.squeeze(-1)

    y_p = _get_validation_predictions(logits_val)
    print(f"*** final epoch score: {calculate_score(y_v, y_p, metric=task):.4f} ***")

    if verbose: 
        embeddings = model.get_embedding(X_val_tensor).detach().cpu().numpy()
        plot_training_results(X_train, X_val, y_t, y_v, y_p, task=task, embed_v=embeddings)

    return model, training_log_df, y_p

def cv_train_nn_model(df: pd.DataFrame, features: list, target: str, model_fn, DEVICE,
                    task: str = "regression", folds: int = 7,
                    num_epochs=100, lr=1e-4, batch_size=2048, save_path=None,
                    TargetTransformer=None, verbose: bool = True):
    """
    """
    mask = df.get("target_mask", pd.Series(True, index=df.index))
    X = df.loc[mask, features]
    y = df.loc[mask, target]
    X_test = df.loc[~mask, features]
    test_tensor = torch.tensor(X_test.values.astype(np.float32), dtype=torch.float32).to(DEVICE)
            
    oof_preds = np.zeros(len(df))
    training_logs, cv_models = [], []
    print(f"Training NN Model...")

    if task.startswith("regression"):
        cv = skl.model_selection.KFold(n_splits=folds, shuffle=True, random_state=69)
    else:
        cv = skl.model_selection.StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in tqdm(enumerate(cv.split(X)), desc="training models", unit="folds"):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = model_fn(X.shape[1])
        verbosisty = False if fold != folds-1 else verbose
        model, log_df, _ = train_and_score_nn_model(X_train, X_val,  y_train, y_val, model, DEVICE,
                                        task=task, verbose=verbosisty, TargetTransformer=TargetTransformer, 
                                        num_epochs=num_epochs, lr=lr, batch_size=batch_size,
                                        save_path=save_path)
 
        with torch.no_grad():
            val_tensor = torch.tensor(X_val.values.astype(np.float32), dtype=torch.float32).to(DEVICE)
            preds = model.predict(val_tensor).squeeze().cpu().numpy()
            oof_preds[val_idx] = preds
            test_preds = model.predict(test_tensor).squeeze().cpu().numpy()
            oof_preds[X_test.index] += test_preds / folds

        training_logs.append(log_df.assign(fold=fold+1))
        cv_models.append(model)

        score = calculate_score(y, oof_preds[y.index], metric=task)
        print(f"***  final fold cv score:  {score:.4f}  ***")

    return cv_models, pd.concat(training_logs), oof_preds

def get_nn_predictions(X, models, batch_size, DEVICE,
                       task: str="regression", n_classes:int=1,
                       get_embed:bool=False):

    def _get_batch(tensor, step):
        start = step * batch_size
        end = min(start + batch_size, len(tensor))
        return tensor[start:end]

    def _decode_logits(logits_val):
        # logits_val is a TORCH tensor on GPU or CPU
        if task.startswith("regression"):
            return logits_val.cpu().numpy().reshape(-1, 1)

        elif "ordinal" in task:
            preds = (logits_val > 0).sum(dim=1)
            return preds.cpu().numpy().reshape(-1, 1)

        elif task.startswith("probability"):
            if logits_val.ndim == 1:
                probs = torch.sigmoid(logits_val)
                return probs.cpu().numpy().reshape(-1, 1)
            else:
                probs = torch.softmax(logits_val, dim=1)
                return probs.cpu().numpy()

        else:  # classification labels
            if logits_val.ndim == 1:
                probs = torch.sigmoid(logits_val)
                preds = (probs > 0.5).long()
                return preds.cpu().numpy().reshape(-1, 1)
            else:
                probs = torch.softmax(logits_val, dim=1)
                preds = probs.argmax(dim=1)
                return preds.cpu().numpy().reshape(-1, 1)

    features = torch.tensor(X.values.astype(np.float32)).to(DEVICE)
    batches = math.ceil(len(features) / batch_size)
    predictions = np.zeros((len(X), n_classes), dtype=np.float32)

    if get_embed:
        embeddings = np.zeros((len(X), n_classes + 16), dtype=np.float32)

    for model in models:
        model_preds = np.zeros((0, n_classes), dtype=np.float32)

        with torch.no_grad():
            for b in range(batches):
                batch = _get_batch(features, b)

                logits = model(batch)
                if logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)

                y_p = _decode_logits(logits)

                model_preds = np.concatenate((model_preds, y_p), axis=0)

            if get_embed:
                emb = model.get_embedding(features).cpu().numpy()
                embeddings[:, n_classes:] += emb

        predictions += model_preds

    predictions /= len(models)

    if get_embed:
        embeddings[:, :n_classes] = predictions
        embeddings /= len(models)
        return embeddings

    return predictions

def submit_nn_predict(X: pd.DataFrame, y: pd.DataFrame, features: list, target:str, 
                      models: list, DEVICE, task: str='regression', TargetTransformer=None, batch_size=4096,
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
    def _plot_target(df, target=target, title=f'predicted {target} distribution', hist=10):
        if pd.api.types.is_float_dtype(df[target]) or (df[target].dtype == int and df[target].nunique() > hist):
            sns.histplot(df[target], 
                         bins = min(df[target].nunique(), 42),  # limit number of bins for large unique value counts
                         kde = True)
        else:
            sns.countplot(data=df, x=target)
        plt.title(title)
        plt.yticks([])
        plt.show()

    y_test = get_nn_predictions(X, models, batch_size, DEVICE, task=task)

    if TargetTransformer is None:
        y_pred = y_test
    else:
        y_pred = TargetTransformer.inverse_transform(y_test.reshape(-1, 1))
    
    submission_df = pd.read_csv(f"{path}/{file}")
    submission_df[target] = y_pred
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)

    if verbose:
        _plot_target(submission_df)

    print("=" * 6, 'save success', "=" * 6, "\n")
    print(f"Predicted target mean: {y_pred.mean():.4f} +/- {y_pred.std():.4f}")

    return submission_df

def get_nn_target_hints(df: pd.DataFrame, features: list, target: str,
                        model, DEVICE, task='regression', index_id:str="1",
                        batch_size=1024, folds=7, num_epochs=20, verbose:bool=True) -> pd.DataFrame:
    """
    """
    mask = df.get("target_mask", pd.Series(True, index=df.index))
    X = df.loc[mask, features]
    y = df.loc[mask, target]
    X_test = df.loc[~mask, features]

    cv = skl.model_selection.KFold(n_splits=folds, shuffle=True, random_state=67)
    if "top_k" in task:
        print("Not ready for multiclass probabilities....yet!")
        #hints = 1 if "top_k" not in task else y.nunique()
    hints = 1 #TODO update get_nn_predictions to support getting multiclass probabilities returned
    y_hint = np.zeros((len(df), hints + 16), dtype=np.float32) # nn_hint + 16 embeddings = 17 columns

    models = []
    for (train_idx, val_idx) in tqdm(cv.split(X, y), total=cv.get_n_splits()):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m, _, _ = train_and_score_nn_model(
            X_train, X_val, y_train, y_val, model, DEVICE, task=task, verbose=False, num_epochs=num_epochs
        )

        preds = get_nn_predictions(
            X_val, [m], batch_size=batch_size, DEVICE=DEVICE, task=task, get_embed=True)
        y_hint[X.index[val_idx], :] = preds
        models.append(m)

    if len(X_test) > 0:
        preds_test = get_nn_predictions(
            X_test, models, batch_size=batch_size, DEVICE=DEVICE, task=task, get_embed=True)
        y_hint[X_test.index, :] = preds_test

    cols = [f"nn_{index_id}_hint_{i}" for i in range(hints)] + [f"nn_{index_id}_embed_{i}" for i in range(16)]
    df_hint = pd.DataFrame(y_hint, index=df.index, columns=cols)
    
    df = pd.concat([df, df_hint], axis=1)
    
    if verbose: _plot_embeddings(df, cols[0], cols[1], target)
    return df

########################################
# Decrement when safe
def impute_using(df: pd.DataFrame, features:List[str], impute_informant: str)-> pd.DataFrame:
    """
    DO NOT USE - FUNCTION REQUIRES VALIDATION
    """
    print("Experimental Function use with CAUTION")
    df['control'] = skl.preprocessing.QuantileTransformer(n_quantiles=100).fit_transform(
        df[[impute_informant]]
    )
    for f in features:
        ds = df.groupby('control')[f].transform(lambda g: g.fillna(g.mean()))
        df[f] = ds
        df[f].fillna(df[f].mean(), inplace=True)
    df.drop('control', axis=1, inplace=True)
    return df

def regression_imputer(df: pd.DataFrame, features: list, split: str, tgt: str,
                       validation_size: float=0.15, impute: bool=True, verbose: bool=False):
    """
    DO NOT USE - FUNCTION REQUIRES VALIDATION
    """
    print("Experimental Function use with CAUTION")
    SEED = 80085
    X_test = df[df[split].eq(True)][features]
    y_test = df[df[split].eq(True)][tgt]

    X = df[df[split].eq(False)][features]
    y = df[df[split].eq(False)][tgt]

    X_train, X_val, y_train, y_val  = skl.model_selection.train_test_split(
        X, y, test_size = validation_size, random_state = SEED)

    imp_model, _ = train_and_score_model(
        X_train, X_val, y_train, y_val, 
        model = lgb.LGBMRegressor(verbose=-1), task="regression_rmse", verbose=verbose
    )

    if impute: df.loc[df[split].eq(True), tgt] = imp_model.predict(X_test)

    return df

def clean_categoricals(df: pd.DataFrame, features: list, 
                       string_length: int=5, fillna:bool=True) -> pd.DataFrame:
    print("Function will be depricated -> use 'clean_strings'")
    return clean_strings(df, features, string_length, fillna)


def get_umap_embeddings(df: pd.DataFrame, features:list, target:str, encode_dim:int = 8, sample:int = 25000, verbose: bool = True) -> pd.DataFrame:
    print("Function will be depricated -> use 'get_embeddings'")
    return get_embeddings(df, features, "UMAP", "umap", sample_size=sample, target=target, encode_dim=encode_dim,
                    verbose=verbose)

def get_pls_embeddings(df: pd.DataFrame, features:list, target:list, col_names:str, 
                   sample_size: float=None, n_components:int=2, verbose=True) -> pd.DataFrame:
    print("Function will be depricated -> use 'get_embeddings'")
    return get_embeddings(df, features, "PLS", col_names, sample_size=sample_size, target=target, encode_dim=n_components,
                    verbose=verbose)

def get_target_labels(df: pd.DataFrame, target: str, targets: list, cuts: int=6, verbose: bool=True):
    print("Function will be depricated target labels generated on data loading")
    return df, targets, target

def plot_null_data(df:pd.DataFrame, features:List[str], verbose:bool=True)->List[str]:
    print("To depricate -> use 'get_features_with_na'")
    return get_features_with_na(df, features, verbose)


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
#import plotly.express as px
#from statsmodels import graphics
#from pyexpat import features