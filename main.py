import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM
from helpers.model import *
from helpers.utils import *


def load_data(filepath):
    return pd.read_csv(filepath)


def get_checkpoints(seed=0, short_exe=False):
    if short_exe:
        return [f'google/multiberts-seed_{seed}-step_{i}k' for i in range(0, 100, 20)]
    return [f'google/multiberts-seed_{seed}-step_{i}k' for i in range(0, 200, 20)] + [
        f'google/multiberts-seed_{seed}-step_{i}k' for i in range(200, 2001, 100)]


def load_model_and_tokenizer(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = BertForMaskedLM.from_pretrained(checkpoint)
    return Model(model, tokenizer)


def predict_masked_token_for_df(df, pt_model, checkpoint):
    step = checkpoint.split('_')[-1]
    pred_col = f'predicted_token{step}'
    label_prob = f'label_prob{step}'
    distractor_prob = f'distractor_prob{step}'
    diff_prob = f'diff_prob{step}'
    top_prob_col = f'top_prob{step}'
    correct_col = f'correct{step}'

    df[[pred_col,label_prob, distractor_prob, diff_prob, top_prob_col]] = df.apply(lambda row: pt_model.predict_masked_token(row), axis=1,
                                            result_type="expand")
    df[correct_col] = (df[pred_col] == df['label1']) | (df[pred_col] == df['label2'])
    return df


def main():
    dataset_niki = {'name': 'niki',
                    'filepath': 'data_processed/niki.csv',
                    'seed': 0,
                    'col1': 'plural_match',
                    'col2': 'length',
                    'humans': {'match': 0.51, 'mismatch': 0.82}}
    dataset_ness = {'name': 'ness',
                    'filepath': 'data_processed/naama.csv',
                    'seed': 0,
                    'col1': 'dependency',
                    'col2': 'match',
                    'humans': {'F-G, true': 0.69, 'F-G, false': 0.75,
                               'S-V, true': 0.76, 'S-V, false': 0.71}}
    datasets = [dataset_niki, dataset_ness]

    for dataset in datasets:
        df = load_data(dataset['filepath'])

        # Initialize the list of accuracies
        seed = dataset['seed']
        list_of_checkpoints = get_checkpoints(seed=seed)
        # Predict the masked token for each checkpoint
        for checkpoint in tqdm(list_of_checkpoints):
            pt_model = load_model_and_tokenizer(checkpoint)
            df = predict_masked_token_for_df(df, pt_model, checkpoint)

        # Plot the accuracies
        plot_by_columns(df, dataset['col1'], dataset['col2'], 'acc', 'Accuracy by Checkpoints', 'Checkpoints',
                        'Accuracy', len(list_of_checkpoints), humans=dataset['humans'])

        plot_by_columns(df, dataset['col1'], dataset['col2'], 'diff_prob',
                        'The difference between label probability and distractor probability over Checkpoints',
                        'Checkpoints', 'Diff probability', len(list_of_checkpoints))

        plot_by_columns(df, dataset['col1'], dataset['col2'], 'entropy', 'Entropy over Checkpoints', 'Checkpoints',
                        'Entropy', len(list_of_checkpoints))

if __name__ == '__main__':
    main()
