import pandas as pd
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
    top_prob_col = f'top_prob{step}'
    correct_col = f'correct{step}'

    df[[pred_col, top_prob_col]] = df.apply(lambda row: pt_model.predict_masked_token(row), axis=1,
                                            result_type="expand")
    df[correct_col] = (df[pred_col] == df['label1']) | (df[pred_col] == df['label2'])
    return df


def calculate_accuracy(df, checkpoint):
    step = checkpoint.split('_')[-1]
    correct_col = f'correct{step}'
    return np.array(df[correct_col]).mean()


def main():
    df = load_data('data_processed/niki.csv')

    accuracies = []
    seed = 0
    list_of_checkpoints = get_checkpoints(seed)

    for checkpoint in list_of_checkpoints:
        pt_model = load_model_and_tokenizer(checkpoint)
        df = predict_masked_token_for_df(df, pt_model, checkpoint)
        accuracy = calculate_accuracy(df, checkpoint)
        print(checkpoint, accuracy)
        accuracies.append(accuracy)

    # plot_by_columns(df, col1, col2, title, x_label, y_label, num_of_checkpoints):
    plot_by_columns(df, 'plural_match', 'length', 'Accuracy by Checkpoints', 'Checkpoints', 'Accuracy',
                    len(list_of_checkpoints))

