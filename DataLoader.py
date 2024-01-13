import pandas as pd

def preprocess(df, tokenizer, seq_len):
    df = df.copy()

    # Remove '@USER' and 'URL' and similar
    df['tweet'] = df['tweet'].str.replace('@USER', '', regex=False)
    df['tweet'] = df['tweet'].str.replace('URL', '', regex=False)
    df['tweet'] = df['tweet'].str.replace('&amp;', '&', regex=False)

    # Remove leading and trailing spaces, and reduce multiple spaces to one
    df['tweet'] = df['tweet'].str.strip().str.replace(r'\s+', ' ', regex=True)

    df['label'] = df['label'].replace({'NOT': 0, 'OFF': 1})

    df['tokens'] = df['tweet'].apply(tokenizer.tokenize)
    df['tokens'] = df['tokens'].apply(lambda x: x[:(seq_len - 2)] if len(x) > seq_len - 2 else x)
    df['tokens'] = df['tokens'].apply(lambda x: [tokenizer.cls_token] + x + [tokenizer.sep_token])
    df['input_ids'] = df['tokens'].apply(tokenizer.convert_tokens_to_ids)
    df['input_mask'] = df['input_ids'].apply(lambda x: [1] * len(x))
    df['segment_ids'] = [[0] * seq_len for _ in range(len(df))]
    df['input_ids'] = df['input_ids'].apply(lambda x: x + [0] * (seq_len - len(x)))
    df['input_mask'] = df['input_mask'].apply(lambda x: x + [0] * (seq_len - len(x)))

    return df

def load_train_data(tokenizer, seq_len):

    df_19_train = pd.read_csv("datasets/OffensEval19/offenseval-training-v1.tsv", sep='\t').rename(columns = {'subtask_a':'label'})

    df_19_test = pd.read_csv("datasets/OffensEval19/testset-levela.tsv", sep='\t')
    df_19_test_label = pd.read_csv("datasets/OffensEval19/labels-levela.csv", names=['id','label'])
    merged_19_test = pd.merge(df_19_test, df_19_test_label, on='id', how='inner')

    train_data = pd.concat([df_19_train[['tweet', 'label']], merged_19_test[['tweet', 'label']]], axis=0)
    formatted_train_data = preprocess(train_data, tokenizer, seq_len)

    return formatted_train_data

def load_test_data(tokenizer, seq_len):

    df_20_test = pd.read_csv("datasets/OffensEval20/test_a_tweets.tsv", sep='\t')
    df_labels = pd.read_csv("datasets/OffensEval20/englishA-goldlabels.csv", names=['id','label'])
    merged_20_test = pd.merge(df_20_test, df_labels, on='id', how='inner')
    if tokenizer == "testing_only":
        merged_20_test['label'] = merged_20_test['label'].replace({'NOT': 0, 'OFF': 1})
        return merged_20_test
    formatted_test_data = preprocess(merged_20_test[['tweet', 'label']], tokenizer, seq_len)

    return formatted_test_data