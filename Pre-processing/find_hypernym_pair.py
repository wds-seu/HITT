import joblib
import xlwt
import xlrd
import sys
import os
from tqdm import tqdm
sys.path.append('..')
from config import DefaultConfig


def pattern_method(parsed_sentence_pkl, candidate_sentences_xlsx):
    sentences = joblib.load(parsed_sentence_pkl)
    patterns = [['以及', '其他'], ['或者', '其他'], ['比如'], ['包括'], ['甚至'], ['是一种']]
    candidate_sentences = []

    for sentence in tqdm(sentences):

        for pattern in patterns:
            pattern_len = len(pattern)
            contain_word_len = 0

            for word in pattern:
                if word in sentence:
                    contain_word_len += 1
                else:
                    break
            if contain_word_len == pattern_len:
                candidate_sentences.append(sentence)
                break
    file = xlwt.Workbook()
    sheet = file.add_sheet(u'sheet1', cell_overwrite_ok=True)

    for i, sentence in enumerate(candidate_sentences):
        sentence_str = ''

        for word in sentence:
            sentence_str += word
        sheet.write(i, 0, sentence_str)
    file.save(candidate_sentences_xlsx)


def syntactic_method(candidate_phrase_total_pkl, candidate_pair_xlsx):
    phrases = joblib.load(candidate_phrase_total_pkl)
    phrases_one_word = []
    phrases_two_word = []
    candidate_pairs = []

    for phrase in phrases:
        if len(phrase) == 1:
            phrases_one_word.append(phrase)
        elif len(phrase) == 2:
            phrases_two_word.append(phrase)

    for phr in phrases_one_word:

        for phr_two in phrases_two_word:
            if phr[0] == phr_two[1]:
                candidate_pairs.append([phr[0], phr_two[0], phr_two[1]])

    file = xlwt.Workbook()
    sheet = file.add_sheet(u'sheet1', cell_overwrite_ok=True)

    for i in range(len(candidate_pairs)):
        sheet.write(i, 0, candidate_pairs[i][0])
        sheet.write(i, 1, candidate_pairs[i][1] + candidate_pairs[i][2])
    file.save(candidate_pair_xlsx)


if __name__ == '__main__':
    opt = DefaultConfig()
    # pattern_method(opt.parse_sentence_pkl, opt.candidate_sentence_xlsx)
    candidate_phrase_total_pkl = opt.daojiao_phrase_dir + str(opt.num_processed) + '.pkl'
    syntactic_method(candidate_phrase_total_pkl, opt.candidate_pairs_xlsx)
