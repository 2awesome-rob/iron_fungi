#functions for use in kaggle tabular data projects
import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib as mpl

import sklearn as skl

def set_globals(seed: int = 67, verbose: bool=True) -> tuple(str, int, mpl.colors.ListedColormap, list):
    """
    -----------
    sets: global variables and configurations for the project
    - seed settings for reproducibility
    - pandas display options
    - seaborn/matplotlib visualization styles
    returns:
    - DEVICE: torch device (cpu or cuda)
    - CORES: number of CPU cores to use for multiprocessing
    - MY_CMAP: custom colormap for visualizations
    - MY_PALETTE: custom color palette for visualizations
    -----------
    requires: numpy, pandas, seaborn, matplotlib, random, multiprocessing.cpu_count
    optional: torch
    -----------
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
    MY_PALETTE = sns.xkcd_palette(['ocean blue', 'gold', 'dull green', 'dusty rose', 'dark lavender', 'carolina blue', 'sunflower', 'lichen', 'blush pink', 'dusty lavender', 'steel grey'])
    MY_CMAP = mpl.colors.ListedColormap(MY_PALETTE)
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
        sns.palplot(MY_PALETTE) 
    
    return DEVICE, CORES, MY_CMAP, MY_PALETTE

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
    
def load_tabular_data(path: str, extra_data: str=None,
                      verbose: bool=True, csv_sep: str=",") -> tuple(pd.DataFrame, list, list, str):
    """
    loads Kaggle type tabular data from path into single DataFrame
    -----------
    assumes:
    - path contains train.csv, test.csv, sample_submission.csv files
    - extra_data is optional "path + file name" of additional training data
        - extra_data must contain same features and targets as train.csv
    -----------
    returns:
    - merged DataFrame for EDA & feature engineering
    - list of training features
        - cleans for consistent pandas friendly feature names
    - list of targets, including column "target_mask"
        - adds "target_mask" for separating test from training data
    - target column name (first target)
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

def get_target_labels(df: pd.DataFrame, target: str, targets: list, cuts:int = 10) -> tuple(pd.DataFrame, list):
    """
    Adds target "label" columns
    Useful for visualizing numeric targets as categorical
    -----------
    if target is categorical (object or category dtype)
        - adds "label" same as target for visualization
    if target is numeric with few unique values (<8)
        - adds "label" reversing the order of the target values (for better visualization)
    if target is numeric with many unique values (>=8)
        - adds "qcut_label" using pd.qcut to create quantile-based bins of the target
        - adds "cut_label" using pd.cut to create equal-width bins of the target
    returns:
    - df with new label columns added
    """
    if df[target].dtype == 'O' or df[target].dtype.name == 'category':
        df["label"] = df[target]
        targets.append("label")
    elif df[target].nunique() < 8:
        df["label"] = XY[target].max() - XY[target]
        targets.append("label")
    else:
        df["qcut_label"] = cuts  - pd.qcut(df[df.target_mask.eq(True)][target], cuts, labels=False)
        df["cut_label"] = cuts  - pd.cut(df[df.target_mask.eq(True)][target], cuts, labels=False)
        df[["qcut_label", "cut_label"]] = df[["qcut_label", "cut_label"]].fillna(-1).astype('int16')
        targets.extend(["qcut_label", "cut_label"])
    return df, targets

def get_transformed_target(df: pd.DataFrame, target: str, 
                           targets: list, name: str="std",
                           TargetTransformer: function=skl.preprocessing.StandardScaler()
                           ) -> tuple(pd.DataFrame, function, list):
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
        - converts to lowercase
        - replaces spaces and special characters
        - trims to specified max length
        - converts to category dtype
    -----------
    returns: updated df with cleaned categorical feature columns
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

def split_training_data(df: pd.DataFrame, features: list, targets: list, validation_size: None
                        )-> tuple:
    """
    splits df into training, validation, and test sets
    -----------
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

def calculate_score(actual, predicted, metric='rmse')-> float:
    """ 
    calculates score based on metric or task
    simplifies calling scikit learn metrics by allowing flexible metric names
    -----------
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

def plot_target_eda(df: pd.DataFrame, target: str, title: str='target distribution', hist: int=20) -> None:
    """
    plots simple target distribution plot
    if target is continuous (float or int with many unique values), plots histogram with KDE
    if target is categorical (object, category, bool, or int with few unique values), plots countplot
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
    for performance, limits the number of features plotted to 20
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
    -----------
    requires: seaborn, matplotlib, pandas, numpy
    """
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

    ### build common cmap for categoricals
    def _set_color_map(order, clrs = 6, sats = 5):
        if len(order) <= len(MY_PALETTE):
            return dict(zip(order, MY_PALETTE[:len(order)]))
        elif len(order) <= clrs * sats:
            new_palette = []
            for j in range(clrs):
                for i in range(sats):
                    new_palette.append(sns.desaturate(MY_PALETTE[j], 1-.2*i))
            return dict(zip(order, new_palette[:len(order)]))
        else:
            cmap = mpl.colormaps['cividis'].resampled(len(order))
            new_palette = [cmap(i / len(order)) for i in range(len(order))]
            return dict(zip(order, new_palette))

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
            color_map = _set_color_map(order)
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
    for performance, limits scatterplots to a sample of the data
    plots:
        - pairwise scatterplots for numeric features
        - kde histograms on the diagonal
        - contour lines on the lower triangle to show density of points in scatterplots
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
                          model: function, task: str="regression", 
                          verbose: bool=False, 
                          TargetTransformer: function=None):
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

def get_feature_importance(X_train, X_val, y_train, y_val, target, verbose = False, task = "regression"):
    """
    gets feature importance by training an LightGBM model
    -----------
    trains a LightGBM model on the training data and evaluates on the validation data
    returns a pandas Series of feature importances sorted in descending order
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
                                                      target=target,
                                                      verbose=False, task=task)
    ds = pd.Series(ModelFeatureImportance.feature_importances_, name="importance", index=X_train.columns)
    ds.sort_values(ascending=False, inplace = True)
    print("=" * 69)
    print(f"  ***  Top feature is: {ds.index[0]}  *** \n")
    ds[:10].plot(kind = 'barh', title = f"Top {min(10, len(ds))} of {len(ds)} Features")
    if verbose:
        print("=" * 69)
        print(f"  Top Features:")
        print(ds.head(12))
        print("=" * 69)
        print(f"  Bottom Features:")
        print("=" * 69)
        print(ds.tail(12))
        print("=" * 69)
        print(f"Zero importance features: {(ds == 0).sum()} of {len(ds.index)}")
    return ds


