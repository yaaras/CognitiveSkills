import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM
from helpers.model import *
from helpers.utils import *


def load_data(filepath):
    return pd.read_csv(filepath)


def get_checkpoints(seed=0):
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
    top_prob_col = f'top_prob{step}'
    correct_col = f'correct{step}'

    df[[pred_col,label_prob,distractor_prob, top_prob_col]] = df.apply(lambda row: pt_model.predict_masked_token(row), axis=1,
                                            result_type="expand")
    df[correct_col] = (df[pred_col] == df['label1']) | (df[pred_col] == df['label2'])
    return df


def calculate_accuracy(df, checkpoint):
    step = checkpoint.split('_')[-1]
    label_prob = f'label_prob{step}'
    distractor_prob = f'distractor_prob{step}'

    return np.array(df[label_prob] - df[distractor_prob]).mean()

def calculate_diff_prob(df, checkpoint):
    step = checkpoint.split('_')[-1]
    correct_col = f'correct{step}'
    return np.array(df[correct_col]).mean()


def main():
    dataset_niki = {'name': 'niki',
                    'filepath': 'data_processed/niki.csv',
                    'seed': 0,
                    'col1': 'plural_match',
                    'col2': 'length',
                    'humans': {'match': 0.51, 'mismatch': 0.77}}

    dataset_ness = {'name': 'ness',
                    'filepath': 'data_processed/naama.csv',
                    'seed': 0,
                    'col1': 'dependency',
                    'col2': 'match',
                    'humans': {'match': 0.51, 'mismatch': 0.77}}

    #datasets = [dataset_niki, dataset_ness]
    datasets = [dataset_niki]


    for dataset in datasets:
        df = load_data(dataset['filepath'])

        # Initialize the list of accuracies
        accuracies = []
        diff_prob_lst = []
        seed = dataset['seed']
        list_of_checkpoints = get_checkpoints(seed)

        # Predict the masked token for each checkpoint
        for checkpoint in tqdm(list_of_checkpoints):
            pt_model = load_model_and_tokenizer(checkpoint)
            df = predict_masked_token_for_df(df, pt_model, checkpoint)
            accuracy = calculate_accuracy(df, checkpoint)
            diff_prob = calculate_diff_prob(df, checkpoint)
            accuracies.append(accuracy)
            diff_prob_lst.append(diff_prob)

        # Plot the accuracies
        plot_by_columns(df, dataset['col1'], dataset['col2'], 'Accuracy by Checkpoints', 'Checkpoints', 'Accuracy',
                        len(list_of_checkpoints))


if __name__ == '__main__':
    main()
