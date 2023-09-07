import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_by_columns(df, col1, col2, metric, title, x_label, y_label, num_of_checkpoints, humans=None):
    # Group the data by the columns
    if humans is None:
        humans = {}
    if metric == 'acc':
        df_groupby = group_by_acc(col1, col2, df)
    else:
        df_groupby = group_by_diff_prob(col1, col2, df)

    steps = np.arange(num_of_checkpoints)

    # Plot the data
    sns.set_theme(style="whitegrid")
    sns.lineplot(x=steps, y=df_groupby.iloc[0], label=df_groupby.index[0])
    sns.lineplot(x=steps, y=df_groupby.iloc[1], label=df_groupby.index[1])
    sns.lineplot(x=steps, y=df_groupby.iloc[2], label=df_groupby.index[2])
    sns.lineplot(x=steps, y=df_groupby.iloc[3], label=df_groupby.index[3])

    if humans:
        sns.lineplot(x=steps, y=humans['correct'], label='humans')

    # Set the title and labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def group_by_acc(col1, col2, df):
    correct_cols = [col for col in df.columns if 'correct' in col]
    return df.groupby([col1, col2])[correct_cols].mean()

def group_by_diff_prob(col1, col2, df):
    diff_prob_cols = [col for col in df.columns if 'diff_prob' in col]
    return df.groupby([col1, col2])[diff_prob_cols].mean()

def group_by_entropy_diff (col1, col2, df):
    diff_prob_cols = [col for col in df.columns if 'diff_prob' in col]
    return df.groupby([col1, col2])[diff_prob_cols].mean()