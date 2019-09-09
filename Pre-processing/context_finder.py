import numpy as np
import joblib
from tqdm import tqdm
import sys
sys.path.append('../')
from config import DefaultConfig
from collections import defaultdict
from collections import Counter


class Global_Context_Finder():
    def __init__(self, candidate_daojiao_pkl, parse_sentence_pkl):
        print('Loading pkl file...')
        self.entitys = joblib.load(candidate_daojiao_pkl)
        self.sentences = joblib.load(parse_sentence_pkl)
        self.entity_len = len(self.entitys)
        self.sentence_len = len(self.sentences)
        print('Done!')

    def is_contain_phrase(self, phrase, sentence):
        """
        Judge a sentence whether contain phrase
        :param phrase:
        :param sentence:
        :return: True / False
        """
        count = 0

        for word in sentence:
            if phrase[count] == word:
                count += 1
            else:
                count = 0
            if count == len(phrase):
                return True

        return False

    def build_sentence_graph(self, save_pkl):
        """
        Build an array which rows represent the sentence and columns represent the entity, if
        entity phrase1 appears in current sentence and correspond position will equal to 1 and
        if the position of the array is equal to 0, it means the current entity does not appear
        in current sentence.
        :return: numpy array about sentence graph and every first number of row indicate whether
        current sentence is contain target phrases.
        """
        arr = []
        count = 0

        for sentence in tqdm(self.sentences):
            curr = []

            for phr_index, phrase in enumerate(self.entitys):
                if self.is_contain_phrase(phrase, sentence):
                    curr.append(phr_index + 1)
                    # print(phrase)
            if curr:
                arr.append(curr)
            else:
                arr.append([0])

        joblib.dump(arr, save_pkl, compress=3)

        for row in arr:
            if row[0] != 0:
                count += 1
        print('The number if sentence that contain target phrase is %d ' % count)

    def load_pairs(self, daojiao_pairs_pkl):
        pairs = joblib.load(daojiao_pairs_pkl)
        positive_number, negative_number = 0, 0
        result, pairs_pool, phrase_pool = [], [], []

        for row in pairs:
            phrase1 = '_'.join(word for word in row[0])
            phrase2 = '_'.join(word for word in row[1])
            compound_phrase = phrase1 + '_' + phrase2
            if phrase1 not in phrase_pool:
                phrase_pool.append(phrase1)
            if phrase2 not in phrase_pool:
                phrase_pool.append(phrase2)
            if compound_phrase not in pairs_pool:
                pairs_pool.append(compound_phrase)
            if row[2] == 0:
                negative_number += 1
            elif row[2] == 1:
                positive_number += 1
            result.append(row)

        add_number = positive_number - negative_number

        while add_number > 0:
            rand_phrase1 = np.random.choice(phrase_pool)
            rand_phrase2 = np.random.choice(phrase_pool)
            rand_compound_phrase = rand_phrase1 + '_' + rand_phrase2
            if rand_phrase1 != rand_phrase2 and rand_compound_phrase not in pairs_pool:
                hype_phrase = rand_phrase1.strip().split('_')
                hypo_phrase = rand_phrase2.strip().split('_')
                result.append([hype_phrase, hypo_phrase, 0])
                add_number -= 1
        print('Total number of pairs is %d ' % (len(result)))

        return result

    def global_context_finder(self, sentence_graph_pkl, daojiao_pairs_pkl, save_dir, min_num=10):
        """
        from sentence graph to find global context, global context means hypernym phrase and hyponym phrase appeared in
        different sentence and the other daojiao entities that appeared in such sentences is the global context.
        :param sentence_graph_pkl:
        :param daojiao_pairs_pkl:
        :param save_dir: save result directory.
        :param min_num: The minimum co-occurrence number of phrases
        :return:
        """
        print('Load data...')
        sentence_graph_list = joblib.load(sentence_graph_pkl)
        daojiao_pairs = self.load_pairs(daojiao_pairs_pkl)
        print('Done!')

        entities_dict = {}
        id_entities = defaultdict(list)
        result = []
        local_count, global_count = 0, 0

        for i, phrase in enumerate(self.entitys):
            id_entities[i+1].extend(phrase)

        for phrase in self.entitys:
            phrase_str = ''.join(word for word in phrase)
            entities_dict[phrase_str] = len(entities_dict) + 1
        daojiao_pairs = tqdm(daojiao_pairs)

        for row in daojiao_pairs:
            daojiao_pairs.set_description("process daojiao pairs!")
            hype_str = ''.join(word for word in row[0])
            hypo_str = ''.join(word for word in row[1])
            hype_num = entities_dict[hype_str] if hype_str in entities_dict else 0
            hypo_num = entities_dict[hypo_str] if hypo_str in entities_dict else 0
            if hype_num == 0 or hypo_num == 0:
                continue
            # local_context_phrase: phrase appear in the sentence that co-occurrence with hype and hypo
            # global context phrase: phrase appear in the sentence that hype and hypo appeared in differently
            # sentence.
            local_context_phrase, global_context_phrase, local_context, global_context = \
                [], [], [], []

            for sen_index, line in enumerate(sentence_graph_list):
                if line[0] == 0:
                    continue
                elif hype_num in line and hypo_num in line:
                    local_context_phrase.extend(line)
                elif hype_num in line or hypo_num in line:
                    global_context_phrase.extend(line)
            if local_context_phrase:
                local_count += 1
                temp = Counter(local_context_phrase).most_common(min_num + 2)
                number, count = 0, 0

                while count < min(min_num, len(temp)) and number < len(temp):
                    curr_num = temp[number][0]
                    if curr_num != hype_num and curr_num != hypo_num:
                        local_context.extend(id_entities[curr_num])
                        count += 1
                    number += 1
            if global_context_phrase:
                global_count += 1
                temp = Counter(global_context_phrase).most_common(min_num + 2)
                number, count = 0, 0

                while count < min(min_num, len(temp)) and number < len(temp):
                    curr_num = temp[number][0]
                    if curr_num != hype_num and curr_num != hypo_num:
                        global_context.extend(id_entities[curr_num])
                        count += 1
                    number += 1

            result.append([row[0], row[1], row[2], local_context, global_context])
        print('The number of pairs with local context is %d ' % local_count)
        print('The number of pairs with global context is %d ' % global_count)

        # result_pkl = save_dir + 'pairs_with_context.pkl'
        # joblib.dump(result, result_pkl, compress=3)


if __name__ == '__main__':
    opt = DefaultConfig()
    daojiao_phrase_pkl = opt.daojiao_phrase_dir + str(opt.num_processed) + '.pkl'
    sentence_graph_pkl = opt.save_dir + 'sentence_graph.pkl'

    global_context_finder = Global_Context_Finder(daojiao_phrase_pkl, opt.parse_sentence_pkl)
    # global_context_finder.build_sentence_graph(sentence_graph_pkl)
    global_context_finder.global_context_finder(sentence_graph_pkl, opt.daojiao_pairs_parsed_pkl, opt.save_dir)