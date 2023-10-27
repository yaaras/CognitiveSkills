import pandas as pd
from transformers import AutoTokenizer, BertForMaskedLM
from helpers.model import *
from helpers.plot import *

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

    df[[pred_col, label_prob, distractor_prob, diff_prob, top_prob_col]] = df.apply(
        lambda row: pt_model.predict_masked_token(row), axis=1,
        result_type="expand")
    df[correct_col] = (df[pred_col] == df['label1']) | (df[pred_col] == df['label2'])
    return df