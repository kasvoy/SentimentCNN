import os
import re
import random

import pandas as pd

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
    
    train_texts = [text.lower() for text in train_texts]
    test_texts  = [text.lower() for text in test_texts]
    
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
    v1_texts = [text.replace('\n', ' ').lower() for text in v1_texts]
    v2_texts = [text.replace('\n', ' ').lower() for text in v2_texts]
    
    return ((v1_texts, v1_labels), (v2_texts, v2_labels))


def load_rotten_split(rotten_path, seed=1):
    
    """
    Utility function for loading the rotten tomatoes critic dataset.
    
    
    Returns:
    
    Training set: Two lists, one for texts, one for labels. Needed for training in this form.
    
    Test set: Multiple test sets, all as lists of (label, text) tuples. Reason for difference in loading is the fact 
    that the test set will be split later in the program, and with tuples the labels are there after a split, with no need of
    reassignment.
    """
    
    df = pd.read_csv(rotten_path).drop_duplicates()
    df = df[['review_type', 'review_content']]

    #getting only non empty reviews
    df = df.loc[df['review_content'].notna()]
    
    #lowercasing all texts
    df['review_content'] = df['review_content'].str.lower()

    #removing the reviews that ask the reader to click somewhere else
    df = df.loc[~df['review_content'].str.contains('full review|click for review|read review')]

    df['review_type'].replace({'Fresh': 1, 'Rotten': 0}, inplace=True)
    df.rename(columns={'review_type': 'positive_review'}, inplace=True)
    
    is_positive_mask = df['positive_review'] == 1
    
    pos_reviews_train_df = df[is_positive_mask].sample(n=10000, random_state=42)    
    neg_reviews_train_df = df[~is_positive_mask].sample(n=10000, random_state=42)
    
    #dropping already sampled texts, so no repeated reviews across test sets.
    df.drop(neg_reviews_train_df.index, inplace=True)
    df.drop(pos_reviews_train_df.index, inplace=True)
    
    #using text.split(" ") on the space to represent a token as it closely resembles the output of a dedicated tokenizer (e.g. within spacy)
    #and counts the number of tokens in this way incomparably faster compared if we had used spacy.
    short_reviews_test_df = df[df['review_content'].apply(lambda x: len(x.split(" ")) < 25)].sample(n=10000, random_state=42)
    df.drop(short_reviews_test_df.index, inplace=True)
    
    random_reviews_test_df = df.sample(n=10000, random_state=42)
    df.drop(random_reviews_test_df.index, inplace=True)
    
    pos_test_df = df[df['positive_review'] == 1].sample(n=1000, random_state=42)
    neg_test_df = df[df['positive_review'] == 0].sample(n=1000, random_state=42)
    

    #(label, text) tuples
    short_reviews_test = list(zip(short_reviews_test_df['positive_review'], short_reviews_test_df['review_content']))
    pos_reviews_train  = list(zip(pos_reviews_train_df['positive_review'], pos_reviews_train_df['review_content']))
    neg_reviews_train  = list(zip(neg_reviews_train_df['positive_review'], neg_reviews_train_df['review_content']))
    random_reviews_test= list(zip(random_reviews_test_df['positive_review'], random_reviews_test_df['review_content']))
    pos_reviews_test   = list(zip(pos_test_df['positive_review'], pos_test_df['review_content']))
    neg_reviews_test   = list(zip(neg_test_df['positive_review'], neg_test_df['review_content']))
    
    
    
    rotten_train = pos_reviews_train + neg_reviews_train
    
    random.seed(seed)
    random.shuffle(rotten_train)
    random.shuffle(short_reviews_test)
    random.shuffle(random_reviews_test)
        
    rotten_train_texts = [text for _, text in rotten_train]
    rotten_train_labels = [label for label,_ in rotten_train]
    

    return (rotten_train_texts, rotten_train_labels, short_reviews_test, random_reviews_test, pos_reviews_test, neg_reviews_test)