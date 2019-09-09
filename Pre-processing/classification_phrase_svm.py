import pandas as pd
import numpy as np
import joblib
import xlrd
import xlwt
import os
import itertools
from tqdm import tqdm
from collections import Counter
from stanfordcorenlp import StanfordCoreNLP
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import sys
sys.path.append("..")
from config import DefaultConfig


def build_vocab(parsed_sentence_pkl, vocabulary_dir, vocabulary_inv_dir, vocab_size=50000):
    sentences = joblib.load(parsed_sentence_pkl)
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_size)]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    vocabulary['<UNK/>'] = len(vocabulary)
    vocabulary_inv.append('<UNK/>')
    joblib.dump(vocabulary, vocabulary_dir, compress=3)
    joblib.dump(vocabulary_inv, vocabulary_inv_dir, compress=3)
    return [vocabulary, vocabulary_inv]


def load_word2vec(word2vec_dir, model_type, vocabulary_inv_dir, embedding_model_dir, num_features=300):
    """
    loads word2vec model
    returns initial weights for embedding layer.

    inputs:
    :param word2vec_dir             # direction of word2vec model
    :param model_type               # baidu encyclopedia / word2vec
    :param vocabulary_inv_dir:      # pkl file contain dict {str:int}
    :param embedding_model_dir:   # dictionary save embedding_weights_pkl
    :param num_features:            # word vector dimensionality
    """
    if model_type == 'word2vec':
        print('Loading existing word2vec model')
        vocabulary_inv = joblib.load(vocabulary_inv_dir)

        # dictionary, where key is word, value is word vectors
        embedding_model = {}

        for line in open(word2vec_dir, 'r'):
            tmp = line.strip().split()
            try:
                word, vec = tmp[0], list(map(float, tmp[1:]))
            except Exception as e:
                continue
            if len(vec) != num_features:
                continue
            if word not in embedding_model:
                embedding_model[word] = np.asarray(vec, dtype='float32')
        # assert(len(embedding_model) == 800000)

    else:
        raise ValueError('Unknown pretrain model type: %s!' % model_type)

    # embedding_weights = [embedding_model[w] if w in embedding_model
    #                      else np.random.uniform(-0.25, 0.25, num_features)
    #                      for w in vocabulary_inv]
    # embedding_weights = np.array(embedding_weights).astype('float32')
    joblib.dump(embedding_model, embedding_model_dir, compress=3)
    return embedding_model


def distributed_representation(embedding_model_dir, candidate_phrase_dir,
                               stanfordcorenle_dir, data_dir, end_index, num_features=300):
    """
    use pre-trained word2vec to generate distributed representation

    :param embedding_model_dir:   # pkl file save weights of embedding
    :param candidate_phrase_dir:    # dictionary of phrase
    :param stanfordcorenle_dir:     # dictionary of stanford-core-nlp package
    :param end_index:               # end of the data with label
    :param data_dir:                # file save train/test/rest/ data
    :param num_features:            # number of features default 300
    :return:                        # train_set [vec, label]
                                    # test_set  [vec, label]
                                    # rest_set  [vec]
    """
    if os.path.exists(embedding_model_dir):
        X = []
        Y = []
        data = xlrd.open_workbook(candidate_phrase_dir)
        table = data.sheets()[0]
        rows = table.nrows
        nlp = StanfordCoreNLP(stanfordcorenle_dir, lang='zh')
        embedding_model = joblib.load(embedding_model_dir)

        for i in tqdm(range(rows)):
            # current_row: save current data of phrase [vec, label]
            current_row = []
            row = table.row_values(i)
            phrase = row[0]

            for word, _ in nlp.pos_tag(phrase):
                current_embeddding = list(embedding_model[word] if word in embedding_model
                                          else np.random.uniform(-0.25, 0.25, num_features))
                if current_row:
                    current_row = [current_embeddding[i] + current_row[i] for i in range(num_features)]
                else:
                    current_row.extend(current_embeddding)
            if i < end_index:
                liuchang = row[3]
                hejidong = row[4]
                daiyawen = row[5]
                value = -1
                if liuchang == hejidong or liuchang == daiyawen:
                    value = liuchang
                elif hejidong == daiyawen:
                    value = hejidong
                if value == 1:
                    Y.append(1)
                else:
                    Y.append(0)
            X.append(current_row)
        train_test_set = X[:end_index]
        rest_set  = X[end_index:]
        print('The number of phrase that related to daojiao field is %d ' % len(train_test_set))
        joblib.dump(train_test_set, os.path.join(data_dir, 'train_test_set.pkl'), compress=3)
        joblib.dump(Y, os.path.join(data_dir, 'label_set.pkl'), compress=3)
        joblib.dump(rest_set, os.path.join(data_dir, 'rest_set.pkl'), compress=3)

        return train_test_set, Y, rest_set
    else:
        print('Load embedding weights failed!')


def svm_model(data_dir):
    X = joblib.load(os.path.join(data_dir, 'train_test_set.pkl'))
    X = np.array(X)
    Y = joblib.load(os.path.join(data_dir, 'label_set.pkl'))
    Y = np.array(Y).reshape(-1, 1)
    unpredicted = joblib.load(os.path.join(data_dir, 'rest_set.pkl'))
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    clf = svm.SVC(C=1, kernel='rbf', degree=3, gamma='auto', decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_train)
    print('Result of training! ')
    print(confusion_matrix(y_train, y_pred))
    y_pred = clf.predict(x_test)
    print('Result of test! ')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    unpredicted_pred = clf.predict(unpredicted)

    return unpredicted_pred, Y.ravel().tolist()


if __name__ == '__main__':
    opt = DefaultConfig()
    # build_vocab(opt.parse_sentence_pkl, opt.vocabulary_pkl, opt.vocabulary_inv_pkl)
    # embedding_weights = load_word2vec(opt.word2vec_dir,
    #                                   'word2vec',
    #                                   opt.vocabulary_inv_pkl,
    #                                   opt.embedding_model_pkl)
    # print(embedding_weights)
    # train_set, test_set, rest_set = distributed_representation(opt.embedding_model_pkl,
    #                                                            opt.candidate_phrase_xlsx_5000,
    #                                                            opt.stanfordcorenlp_dir,
    #                                                            opt.data_dir,
    #                                                            5000)
    # predicted_label, exist_label = svm_model(opt.data_dir)
    # data = xlrd.open_workbook(opt.candidate_phrase_xlsx_5000)
    # table = data.sheets()[0]
    # rows = table.nrows
    # process_number = 5000
    # candidate_phrase = []
    #
    # for i in tqdm(range(rows)):
    #     row = table.row_values(i)
    #     if i < process_number and exist_label[i] == 1:
    #         candidate_phrase.append(row[0])
    #     elif i >= process_number and predicted_label[i-process_number] == 1:
    #         candidate_phrase.append(row[0])
    # file = xlwt.Workbook()
    # sheet = file.add_sheet(u'sheet1', cell_overwrite_ok=True)
    #
    # for i in range(len(candidate_phrase)):
    #     sheet.write(i, 0, candidate_phrase[i])
    # file.save(os.path.join(opt.data_dir, 'candidate_phrase_total_' + str(process_number) + '.xlsx'))
    label_dir = opt.data_dir + 'label_set.pkl'
    labels = joblib.load(label_dir)
    count = 0
    for label in labels:
        if label == 1:
            count += 1
    print('The number of phrase that related to daojiao field is %d ' % count)