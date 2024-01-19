import os
import re
import random

import pandas as pd
import numpy as np

IMDB_DATA_PATH        = "data/aclImdb/"
POLARITY_v1_DATA_PATH = "data/rt-polaritydata/rt-polaritydata/"
POLARITY_v2_DATA_PATH = "data/review_polarity/txt_sentoken"
ROTTEN_PATH           = "data/rotten/rotten_tomatoes_critic_reviews.csv"

def load_inidv_dataset(set_path: str) -> tuple:
    texts = []
    labels = []
    
    for label in ['pos', 'neg']:
        cat_path = os.path.join(set_path, label)
        for file_name in os.listdir(cat_path):
            file_path = os.path.join(cat_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                texts.append(text)
            labels.append(0 if label=='neg' else 1)
            
    return (texts, labels)


def load_imdb_dataset(train_path: str, test_path: str, seed=1) -> tuple:
    train_texts = []
    train_labels = []
    
    (train_texts, train_labels) = load_inidv_dataset(set_path=train_path)
    (test_texts, test_labels)   = load_inidv_dataset(set_path=test_path)

    random.seed(seed)
    random.shuffle(train_texts)
    random.shuffle(test_texts)
    
    random.seed(seed)
    random.shuffle(train_labels)
    random.shuffle(test_labels)
    
    #removing html tags present in the data
    train_texts = [re.sub('<.*?>', '', text) for text in train_texts]
    test_texts  = [re.sub('<.*?>', '', text) for text in test_texts]
    
    return ((train_texts, train_labels), (test_texts, test_labels))


def load_polarity(v1_path, v2_path):
    
    v1_file_names = os.listdir(v1_path)
    
    v1_texts = []
    v1_labels = []
    
    for file_name in v1_file_names:
        file_path = os.path.join(v1_path, file_name)
        label = (0 if 'neg' in file_name else 1)

        with open(file_path, 'r', errors='ignore') as file:
            text = file.read()

            for snippet in text.splitlines():
                v1_texts.append(snippet)
                v1_labels.append(label)

    (v2_texts, v2_labels) = load_inidv_dataset(set_path=v2_path)
    
    #removing newline tags present in the polarity texts
    v1_texts = [text.replace('\n', ' ') for text in v1_texts]
    v2_texts = [text.replace('\n', ' ') for text in v2_texts]
    
    return ((v1_texts, v1_labels), (v2_texts, v2_labels))


def load_rotten(rotten_path, seed=1):
    df = pd.read_csv(ROTTEN_PATH).drop_duplicates()
    df = df[['review_type', 'review_content']]

    #getting only non empty reviews
    df = df.loc[df['review_content'].notna()]

    #removing the reviews that ask the reader to click somewhere else
    df = df.loc[~df['review_content'].str.contains('full review|click for review|read review')]

    df['review_type'].replace({'Fresh': 1, 'Rotten': 0}, inplace=True)
    df.rename(columns={'review_type': 'positive_review'}, inplace=True)

    is_positive = df['positive_review'] == 1
    pos_df = df.loc[is_positive]
    neg_df = df.loc[~is_positive]

    pos_texts = pos_df['review_content'].tolist()
    neg_texts = neg_df['review_content'].tolist()

    pos_25 = random.sample(pos_texts, 25000)
    neg_25 = random.sample(neg_texts, 25000)

    train_pos = list(zip(pos_25[:12500], [1 for _ in range(12500)]))
    test_pos  = list(zip(pos_25[12500:], [1 for _ in range(12500)]))
    train_neg = list(zip(neg_25[:12500], [0 for _ in range(12500)]))
    test_neg  = list(zip(neg_25[12500:], [0 for _ in range(12500)]))

    train_set = train_pos + train_neg
    test_set  = test_pos + test_neg

    train_texts, train_labels = zip(*train_set)
    test_texts, test_labels = zip(*test_set)

    train_texts = list(train_texts)
    train_labels = list(train_labels)
    test_texts = list(test_texts)
    test_labels = list(test_labels)

    random.seed(seed)
    random.shuffle(train_texts)
    random.shuffle(test_texts)

    random.seed(seed)
    random.shuffle(train_labels)
    random.shuffle(test_labels)
    
    return ((train_texts, train_labels), (test_texts, test_labels))


def display_dataset_info(texts, labels, name):
    print(f"Dataset: {name}.")
    print(f"Total number of samples: {len(texts)}")
    print(f"Positive reviews total: {sum(label==1 for label in labels)}")
    print(f"Negative reviews total: {sum(label==0 for label in labels)}")
    

