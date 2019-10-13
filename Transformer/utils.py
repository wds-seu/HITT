# utils.py

import torch
from torchtext import data
import spacy
import pandas as pd
import numpy as np
import joblib
from torchtext.vocab import Vectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.strip()[-1])

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        with open(filename, 'r') as datafile:     
            data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: self.parse_label(x[0]), data))

        full_df = pd.DataFrame({"text":data_text, "label":data_label})
        return full_df

    def get_my_pandas_df(self, filename, context_flag):
        """
        Load data from pkl file
        :param filename:
        :param context_flag:    # 0: barely include pairs
                                # 1: include pairs and local context
                                # 2: include pairs and global context
                                # 3: include pairs, local context and global context
        :return:
        """
        pairs = joblib.load(filename)
        if context_flag == 0:
            data_pairs = [row[0] + row[1] for row in pairs]
        elif context_flag == 1:
            data_pairs = [row[0] + row[1] + row[3] for row in pairs]
        elif context_flag == 2:
            data_pairs = [row[0] + row[1] + row[4] for row in pairs]
        elif context_flag == 3:
            data_pairs = [row[0] + row[1] + row[3] + row[4] for row in pairs]
        data_label = [row[2] for row in pairs]

        train_text, val_text, train_label, val_label = train_test_split(data_pairs, data_label,
                                                                        test_size=0.20,
                                                                        shuffle=True)
        train_text, test_text, train_label, test_label = train_test_split(train_text, train_label,
                                                                          test_size=0.25,
                                                                          shuffle=True)
        train_df = pd.DataFrame({"text": train_text, "label": train_label})
        test_df = pd.DataFrame({"text": test_text, "label": test_label})
        val_df = pd.DataFrame({"text": val_text, "label": val_label})
        return train_df, test_df, val_df
    
    def load_data(self, train_file, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        NLP = spacy.load('en')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        
        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text",TEXT),("label",LABEL)]
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        
        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data.split(split_ratio=0.8)
        
        TEXT.build_vocab(train_data)
        self.vocab = TEXT.vocab
        
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))

    def load_my_data(self, word_embedding_pkl, pairs_pkl):
        """
        Loads the data from file
        :param word_embedding_pkl: absolute path to word_embeddings {Glove/Word2Vec}
        :param pairs_pkl:       # pkl file save data
        :return:
        """
        tokenizer = lambda text: [x for x in text]

        TEXT = data.Field(sequential=True, tokenize=tokenizer, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT), ("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df, test_df, val_df = self.get_my_pandas_df(pairs_pkl, self.config.context_flag)

        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)

        val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
        val_data = data.Dataset(val_examples, datafields)

        TEXT.build_vocab(train_data, vectors=Vectors(name=word_embedding_pkl))
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab
        # TEXT.build_vocab(train_data)
        # self.vocab = TEXT.vocab

        self.train_iterator = data.BucketIterator(
            train_data,
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)

        print('Loaded %d training example' % len(train_data))
        print('Loaded %d test example ' % len(test_data))
        print('Loaded %d validation examples' % len(val_data))


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        if torch.cuda.is_available():
            x = batch.text.cuda()
        else:
            x = batch.text
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1]
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score