import os
import joblib
import codecs
import argparse
import math
import pandas as pd
import numpy as np
from random import choice
from config import *
from parallelize import parallelize_with_results
from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
from nltk.tree import Tree
from collections import defaultdict


class PreProcessing:
    """
    Pre-processing the raw data of text and get the data of tsv file of input.
    Step 1: remove the noise data
            the sentence without proper ending and the length of current sentence
            smalled than certain value will be removed
    Step 2: use standford core nlp package to parse noun phrase in the sentence, and calculate
            the TF-IDF value of each noun phrase in the total documents
    Step 3: remove the duplicate sentences and arrange them in the order of paragraph order.
    Step 4: count the noun phrase in the sentence and sort them in the order of frequency and
            TF-IDF value of each noun phrase.
    Step 5: manually label the top 20,000 noun phrase to distinguish whether they are belong to
            the domain we are interested.
    """
    def __init__(self, stanfordcorenlp_dir=''):
        """
        Loading stanford core nlp package to extract the noun phrase of sentence.
        :param stanfordcorenlp_dir: file contain Stanford corenlp toolkit
        """
        if os.path.exists(stanfordcorenlp_dir):
            print('Loading stanford core nlp package...')
            self.nlp = StanfordCoreNLP(stanfordcorenlp_dir, lang='zh')
            print('Finished!')
        else:
            self.nlp = None
        self.entity_dict = defaultdict(list)

    def close_service(self):
        if self.nlp is not None:
            self.nlp.close()

    def construct_sentence_graph(self, total_sentence_txt):
        """
        construct sentence graph and calculate the frequency of each noun phrase in the total
        sentences. Finally the result will save into entity_id_count_pkl file in the style of
        {phrase_str: [id, count]}. The list of sentences in the style of
        [[phrase_id1, phrase_id2, ...]...] will be saved into sentence_graph.pkl file.
        :param total_sentence_txt:
        :return:
        """
        file = codecs.open(total_sentence_txt, 'r', errors='ignore', encoding='utf-8')
        sentences = tqdm(file.readlines())
        sentences_list = []

        for sentence in sentences:
            curr_np_list = []
            try:
                sentence_parse = self.nlp.parse(sentence)

                for node in Tree.fromstring(sentence_parse).subtrees():
                    if node.label() == 'NP':
                        node_str = ''.join(word for word in node.leaves())
                        curr_np_list.append(node_str)
            except Exception as e:
                pass
            if curr_np_list:
                correct_np_list = []

                for phrase in curr_np_list:
                    flag = True

                    for char in phrase:
                        if u'\u4e00' > char or u'\u9fff' < char:
                            flag = False
                            break
                    if flag is True and phrase not in self.entity_dict.keys():
                        self.entity_dict[phrase].append(len(self.entity_dict) + 1)
                        self.entity_dict[phrase].append(1)
                        correct_np_list.append(self.entity_dict[phrase][0])
                    elif flag is True:
                        self.entity_dict[phrase][1] += 1
                        correct_np_list.append(self.entity_dict[phrase][0])
                if correct_np_list:
                    sentences_list.append(correct_np_list)
                else:
                    sentences_list.append([0])
            else:
                sentences_list.append([0])
        # for key, value in self.entity_dict.items():
        #     print(key)
        #     print(value)
        # print(sentences_list)
        row_data_file = total_sentence_txt.rsplit('/', 1)[0]
        sentences_pkl = os.path.join(row_data_file, 'sentence_graph.pkl')
        entity_id_count_pkl = os.path.join(row_data_file, 'entity_id_count_pkl')
        print(sentences_pkl)
        print('The length of sentence graph is: ', len(sentences_list))
        print(entity_id_count_pkl)
        print('The size of entity is: ', len(self.entity_dict))
        # joblib.dump(self.entity_dict, entity_id_count_pkl, compress=3)
        # joblib.dump(sentences_list, sentences_pkl, compress=3)

    def compute_tf_df(self, file_list, entity_id_count_pkl):
        """
        compute the tf_idf of every noun phrase
        :param file_list:               directory file save row documents
        :param entity_id_count_pkl:     pkl file in the style of
                                        {phrase_str: [id, tf]}
        :return:                        pkl file in the style of
                                        {phrase_str, [id, tf, df]}
        """
        entity_id_count_dict = joblib.load(entity_id_count_pkl)
        if len(self.entity_dict) == 0:
            self.entity_dict = entity_id_count_dict

        for key, _ in entity_id_count_dict.items():
            if len(entity_id_count_dict[key]) == 2:
                entity_id_count_dict[key].append(0)
            elif len(entity_id_count_dict[key]) == 3:
                entity_id_count_dict[key][2] = 0
            else:
                print('Something wrong with entity_id_count_pkl file!')

        for file in file_list:
            if os.path.exists(file):
                print('Current directory is: ', file)
                document_count = 0
                results = []
                document_list = [os.path.join(file, document) for document in os.listdir(file)]
                results.extend(list(parallelize_with_results(self.extract_from_document, document_list, workers=12)))

                for result in results:

                    if result:
                        document_count += 1

                        for phrase in result:
                            entity_id_count_dict[phrase][2] += 1
                print('Total number of documents in current directory is ', len(os.listdir(file)))
                print('Valid number of documents in current directory is ', document_count)
        self.entity_dict = entity_id_count_dict
        # joblib.dump(entity_id_count_dict, entity_id_count_pkl, compress=3)

    def extract_from_document(self, document_dir):
        """
        Extract noun phrase from document_dir
        :param document_dir:
        :return:
        """
        result = []
        sentences = []
        with codecs.open(document_dir, 'r+', encoding='utf-8', errors='ignore') as file:
            sen = ''

            for line in file.readlines():
                line = line.lstrip().rstrip()

                for word in line:
                    if word not in ['。', '！', '？']:
                        sen += word
                    else:
                        sen += word
                        if len(sen) > 8:
                            sentences.append(sen.strip() + '\n')
                        sen = ''
                if sen != '' and sen[-1] in ['：', '；', '，', '、']:
                    continue
                else:
                    sen = ''

            for sentence in sentences:
                try:
                    sentence_parse = self.nlp.parse(sentence)

                    for node in Tree.fromstring(sentence_parse).subtrees():
                        if node.label() == 'NP':
                            phrase_str = ''.join(word for word in node.leaves())
                            if phrase_str not in result and phrase_str in self.entity_dict.keys():
                                result.append(phrase_str)
                except Exception as e:
                    pass

        return result

    def compute_tfidf(self, doc_count, entity_id_count_pkl):
        """
        compute tf idf value of each noun phrase in corpus.
        :param doc_count:               valid numbers of document
        :param entity_id_count_pkl:     entity dict file in the style of
                                        {phrase_str: [id, tf, df]}
        :return:                        entity dict file in the style of
                                        {phrase_str: [id, tf, df, tf_idf]}
        """
        entity_id_count_dict = joblib.load(entity_id_count_pkl)
        entity_count = len(entity_id_count_dict)
        id_entity_dict = {}
        id_count_list = []

        for key, value in entity_id_count_dict.items():
            if len(value) == 3:
                curr_tf_idf = math.log(doc_count / (value[2] + 1)) * (value[1] / entity_count)
                entity_id_count_dict[key].append(curr_tf_idf)
                id_count_list.append(entity_id_count_dict[key])
                id_entity_dict[value[0]] = key
        id_count_list.sort(key=lambda id_count_list: id_count_list[3], reverse=True)

        # for value in id_count_list[:5000]:
        #     print(id_entity_dict[value[0]])
        #     print(value)
        # joblib.dump(entity_id_count_dict, entity_id_count_pkl, compress=3)

    def extract_context_sentences(self, total_sentence_txt, row_directory, entity_id_count_pkl):
        """
        Extract sentence which contian certain noun phrase and save them in entity_id_count_pkl file
        in the style of {phrase_str: [id, tf, df, tf-idf, randomly k context sentences]}
        :param total_sentence_txt:
        :param row_directory:
        :param entity_id_count_pkl:
        :return: entity_id_count_pkl in the style of
                {phrase_Str: [id, tf, df, tf-idf, randomly k context sentences]}
                and save [phrase_str, tf-idf, randomly k context sentences] into csv file
        """
        file = codecs.open(total_sentence_txt, 'r', errors='ignore', encoding='utf-8')
        row_sentence_list = file.readlines()
        sentence_graph_pkl = os.path.join(row_directory, 'sentence_graph.pkl')
        sentence_graph_list = joblib.load(sentence_graph_pkl)
        if len(row_sentence_list) != len(sentence_graph_list):
            print('Something wrong happened in the length of row sentences!')
            return
        entity_id_count_dict = joblib.load(entity_id_count_pkl)
        id_entity_dict = {}
        id_count_list = []

        for key, value in entity_id_count_dict.items():
            if len(value) == 4:
                id_count_list.append(entity_id_count_dict[key])
                id_entity_dict[value[0]] = key
        id_count_list.sort(key=lambda id_count_list: id_count_list[3], reverse=True)

        for value in tqdm(id_count_list[:10000]):
            index_list = []
            sentences_str = ''

            for index, sen in enumerate(sentence_graph_list):
                if value[0] in sen:
                    index_list.append(index)
            if len(index_list) >= 3:
                i = 0
                total_count = 10
                temp_sent_list = []

                while i < 3 and total_count > 0:
                    curr_sent_num = choice(index_list)
                    if curr_sent_num not in temp_sent_list and len(row_sentence_list[curr_sent_num]) < 100:
                        temp_sent_list.append(curr_sent_num)
                        i += 1
                    total_count -= 1
                index_list = temp_sent_list
            elif 1 <= len(index_list) < 3:
                print('Current size of sentences corresponding to {} is less than 3!'.format(value[0]))
            else:
                index_list.append(-1)
            # print(id_entity_dict[value[0]])
            # print(entity_id_count_dict[id_entity_dict[value[0]]])

            for index, num in enumerate(index_list):
                if 0 <= num < len(row_sentence_list):
                    sentences_str += row_sentence_list[num].strip('\n') + '\n'
                else:
                    sentences_str += id_entity_dict[value[0]]
            sentences_str = sentences_str.strip('\n')
            # if len(entity_id_count_dict[id_entity_dict[value[0]]]) == 4:
            #     # print(sentences_str)
            #     entity_id_count_dict[id_entity_dict[value[0]]].append(sentences_str)
            value.append(sentences_str)

        # save list into csv file.
        print('Save entity sentences file into csv file')
        entity_sentences_csv = os.path.join(row_directory, 'entity_sentences.csv')
        result = list()

        for value in id_count_list[:10000]:
            if len(value) == 5:
                result.append([id_entity_dict[value[0]], str(value[3]), value[4]])
                print(id_entity_dict[value[0]])
                print(value[3])
                print(value[4])
        result = np.array(result, dtype=str)
        df = pd.DataFrame(result)
        df.to_csv(entity_sentences_csv, sep='\t', index=False, header=False)
        print('Done!')


def test():
    opt = Config()
    preprocessing = PreProcessing(opt.stanford_corenlp_dir)
    # preprocessing.construct_sentence_graph(opt.total_sentence_txt)
    # file_list = [os.path.join(opt.row_directory, file)
    #              for file in ['66law',
    #                           'chinacourt',
    #                           'chinalaw',
    #                           'findlaw',
    #                           'globallaw',
    #                           'zhengce']]
    # preprocessing.compute_tf_df(file_list, opt.entity_pkl)
    # doc_count = 51951
    # preprocessing.compute_tfidf(doc_count, opt.entity_pkl)
    preprocessing.extract_context_sentences(opt.total_sentence_txt, opt.row_directory, opt.entity_pkl)
    preprocessing.close_service()


if __name__ == '__main__':
    test()



