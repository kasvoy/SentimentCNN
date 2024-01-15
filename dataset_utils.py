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
    
    #remove html tags from the texts
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

   
    (v2_texts, v2_labels) = load_inidv_dataset(set_path=POLARITY_v2_DATA_PATH)
    
    return ((v1_texts, v1_labels), (v2_texts, v2_labels))


def load_rotten(rotten_path, seed=1):
    df = pd.read_csv(rotten_path).drop_duplicates()
    df = df[['review_type', 'review_content']]
    
    #getting only non empty reviews
    df = df.loc[df['review_content'].notna()]
    
    #removing the reviews that ask the reader to click somewhere else
    df = df.loc[~df['review_content'].str.contains('full review|click for review|read review')]
    
    df['review_type'].replace({'Fresh': 1, 'Rotten': 0}, inplace=True)
    df.rename(columns={'review_type': 'positive_review'}, inplace=True)
    
    rotten_texts  = df['review_content'].to_list()
    rotten_labels = df['positive_review'].to_list()
    
    random.seed(seed)
    random.shuffle(rotten_texts)
    random.shuffle(rotten_labels)
    
    return (rotten_texts, rotten_labels)