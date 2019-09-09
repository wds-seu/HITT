# -*-coding:utf-8-*-
import os

class DefaultConfig(object):
    """ Default parameters and settings """

    # high parameter:
    num_fetures = 300
    num_processed = 5000

    # dictionary that save original data
    original_data_dir = '../data/total_data.txt'

    # phrase save dictionary
    parse_sentence_pkl = '../data/parse_sentence.pkl'
    vocabulary_save_dir = '../data/vocabulary.txt'
    vocabulary_pkl = '../data/vocabulary.pkl'
    vocabulary_inv_pkl = '../data/vocabulary_inv.pkl'
    parse_phrase_txt = '../data/phrase.txt'
    parse_phrase_pkl = '../data/phrase.pkl'
    phrase_one_or_two_csv = '../data/phrase_one_or_two_count.csv'
    phrase_multi_csv = '../data/phrase_multi_count.csv'
    phrase_csv = '../data/phrase_csv.csv'
    filter_phrase_pkl = '../data/filter_phrase.pkl'
    filter_phrase_txt = '../data/filter_phrase.txt'
    candidate_phrase_xlsx_5000 = '../data/total-5000.xlsx'
    candidate_phrase_csv_5000 = '../data/candidate-5000.csv'
    sentence_phrase_split_csv = '../data/sentence_phrase_split.csv'
    data_dir = '../data/model_data/'
    daojiao_phrase_dir = '../data/model_data/candidate_phrase_total_'
    candidate_pairs_xlsx = '../data/model_data/candidate_pairs.xlsx'
    candidate_pairs_daojiao_xlsx = '../data/model_data/candidate_pairs_daojiao.xlsx'
    previous_daojiao_xlsx = '../data/model_data/previous_daojiao_pairs.xlsx'
    daojiao_pairs_pkl = '../data/model_data/daojiao_pairs.pkl'
    daojiao_pairs_xlsx = '../data/model_data/daojiao_pairs.xlsx'
    daojiao_pairs_parsed_pkl = '../data/model_data/daojiao_pairs_parsed.pkl'

    # demo file dictionary
    demo_original_data_dir = '../data/demo/demo_total_data.txt'
    demo_parse_sentence_pkl = '../data/demo/demo_parse_sentence.pkl'
    demo_parse_phrase_txt = '../data/demo/demo_parse_phrase_txt'
    demo_parse_phrase_pkl = '../data/demo/demo_parse_phrase_pkl'
    demo_phrase_one_or_two_csv = '../data/demo/demo_phrase_one_or_two_count.csv'
    demo_phrase_multi_csv = '../data/demo/demo_phrase_multi_count.csv'
    demo_filter_phrase_pkl = '../data/demo/demo_filter_phrase.pkl'
    demo_filter_phrase_txt = '../data/demo/demo_filter_phrase.txt'

    # dictionary that save chinese stanford core nlp modules
    stanfordcorenlp_dir = r'../stanford-corenlp-full-2018-10-05/'

    # dictionary that save embedding files:
    word2vec_dir = '../data/sgns.baidubaike.bigram-char'
    embedding_model_pkl = '../data/embedding_model.pkl'

    # candidate sentences file:
    candidate_sentence_xlsx = '../data/model_data/candidate_sentence.xlsx'

    # input model file:
    input_model_vocabulary_pkl = 'data/input_data/vocabulary.pkl'
    padded_phrase_pkl = 'data/input_data/padded_phrase.pkl'
    save_dir = '../data/input_data/'
    local_context_dir = '../data/input_data/pairs_with_local_context'
