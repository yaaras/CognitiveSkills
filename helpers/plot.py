import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import differential_entropy, norm


def plot_by_columns(df, col1, col2, metric, title, x_label, y_label, num_of_checkpoints, humans=None):
    # Group the data by the columns
    if humans is None:
        humans = {}
    if metric == 'acc':
        df_groupby = group_by_acc(col1, col2, df)
    elif metric == 'diff_prob':
        df_groupby = group_by_diff_prob(col1, col2, df)
    elif metric == 'entropy':
        df_groupby = group_by_entropy_diff(col1, col2, df)
    else:
        raise ValueError('metric not implemented')
    steps = np.arange(num_of_checkpoints)

    # Plot the data
    sns.set_theme(style="whitegrid")
    sns.lineplot(x=steps, y=df_groupby.iloc[0], label=df_groupby.index[0])
    sns.lineplot(x=steps, y=df_groupby.iloc[1], label=df_groupby.index[1])
    sns.lineplot(x=steps, y=df_groupby.iloc[2], label=df_groupby.index[2])
    sns.lineplot(x=steps, y=df_groupby.iloc[3], label=df_groupby.index[3])

    if humans:
        for key, value in humans.items():
            sns.lineplot(x=[0, num_of_checkpoints - 1], y=value, linestyle='--', label=f"Humans-{key}")

    # Set the title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def get_grouped_data(df, col1, col2, metric):
    if metric == 'acc':
        return group_by_acc(col1, col2, df)
    elif metric == 'diff_prob':
        return group_by_diff_prob(col1, col2, df)
    elif metric == 'entropy':
        return group_by_entropy_diff(col1, col2, df)
    else:
        raise ValueError('metric not implemented')


def plot_on_axis(ax, df_groupby, steps, humans):
    sns.lineplot(x=steps, y=df_groupby.iloc[0], label=df_groupby.index[0], ax=ax)
    sns.lineplot(x=steps, y=df_groupby.iloc[1], label=df_groupby.index[1], ax=ax)
    sns.lineplot(x=steps, y=df_groupby.iloc[2], label=df_groupby.index[2], ax=ax)
    sns.lineplot(x=steps, y=df_groupby.iloc[3], label=df_groupby.index[3], ax=ax)

    if humans:
        for key, value in humans.items():
            sns.lineplot(x=[0, len(steps) - 1], y=value, linestyle='--', label=f"Humans-{key}", ax=ax)


def plot_on_subplot(df, col1, col2, metric, title, x_label, y_label, num_of_checkpoints, ax, humans=None, legend=False):
    if humans is None:
        humans = {}

    df_groupby = get_grouped_data(df, col1, col2, metric)
    steps = np.arange(num_of_checkpoints)
    colors = ['r', 'b', 'g', 'y']

    sns.lineplot(x=steps, y=df_groupby.iloc[0], label=df_groupby.index[0], ax=ax, legend=legend, color=colors[0])
    sns.lineplot(x=steps, y=df_groupby.iloc[1], label=df_groupby.index[1], ax=ax, legend=legend, color=colors[1])
    sns.lineplot(x=steps, y=df_groupby.iloc[2], label=df_groupby.index[2], ax=ax, legend=legend, color=colors[2])
    sns.lineplot(x=steps, y=df_groupby.iloc[3], label=df_groupby.index[3], ax=ax, legend=legend, color=colors[3])

    if humans:
        i = 0
        for key, value in humans.items():
            sns.lineplot(x=[0, len(steps) - 1], y=value, linestyle='--', label=f"Humans-{key}", ax=ax, legend=legend,
                         color=colors[i])
            i += 1

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def calculate_diff_entropy(column):
    return differential_entropy(column)


def df_diff_entropy(df):
    return df.apply(calculate_diff_entropy)


def calculate_entropy(column):
    df = pd.DataFrame({'x': column.values})
    df['x_discretized'] = pd.cut(df['x'], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                 include_lowest=True)
    column_disc = df['x_discretized']
    p = column_disc.value_counts() / len(column_disc)
    p[p == 0] = 1

    # Calculate the entropy
    entropy = -np.sum(p * np.log10(p))
    return entropy


def df_entropy(df):
    return df.apply(calculate_entropy)


def group_by_acc(col1, col2, df):
    correct_cols = [col for col in df.columns if 'correct' in col]
    return df.groupby([col1, col2])[correct_cols].mean()


def group_by_diff_prob(col1, col2, df):
    diff_prob_cols = [col for col in df.columns if 'diff_prob' in col]
    return df.groupby([col1, col2])[diff_prob_cols].mean()


def group_by_entropy_diff(col1, col2, df):
    diff_prob_cols = [col for col in df.columns if 'label_prob' in col]
    grouped = df.groupby([col1, col2])
    entropy_results = grouped[diff_prob_cols].apply(df_entropy)
    # mean_entropy_results = entropy_results.groupby(level=[0, 1], axis=0)

    return entropy_results
