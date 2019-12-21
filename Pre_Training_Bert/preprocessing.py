import argparse
import joblib
import os
import pandas as pd
import random
from tqdm import tqdm


def entity_processing(entity_pkl):
    """
    Load entity pkl file and return the dict of {entity : id}
    :param entity_pkl:  pkl file contains entity
    :return:            dict {entity : id}
    """
    entity_list = joblib.load(entity_pkl)
    entity_dict = {}

    for phrase in entity_list:
        phrase_str = ''.join(word for word in phrase)
        entity_dict[phrase_str] = len(entity_dict) + 1

    return entity_dict


def main():
    """
    Convert entity pairs with context into sentences pairs which will fed into
    bert model
    :return:
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--input_pairs_path", type=str, required=True,
                        help="Path of the input pkl file which in the form of "
                             "[hype, hypo, type]")
    parser.add_argument("--sentence_graph_pkl", type=str, required=True,
                        help="Pkl file of sentence graph.")
    parser.add_argument("--entity_pkl", type=str, required=True,
                        help="Pkl file of entities")
    parser.add_argument("--sentence_path", type=str, required=True,
                        help="Path of the sentence pkl file.")
    parser.add_argument("--output_pairs_sentences_dir", type=str, required=True,
                        help="Dictionary of pairs with sentences.")
    args = parser.parse_args()

    print("Loading data!")
    if args.input_pairs_path is not None and os.path.exists(args.input_pairs_path):
        pairs_contexts = joblib.load(args.input_pairs_path)
    if args.sentence_graph_pkl is not None and os.path.exists(args.sentence_graph_pkl):
        sentences_graph = joblib.load(args.sentence_graph_pkl)
    if args.entity_pkl is not None and os.path.exists(args.entity_pkl):
        entity_dict = entity_processing(args.entity_pkl)
    if args.sentence_path is not None and os.path.exists(args.sentence_path):
        sentences = joblib.load(args.sentence_path)
    if args.output_pairs_sentences_dir is not None and \
        os.path.exists(args.output_pairs_sentences_dir):
        output_pairs_sentences_dir = args.output_pairs_sentences_dir
    print("The size of input pairs is: ", len(pairs_contexts))
    print("The size of entity is: ", len(entity_dict))
    print("The size of sentences is: ", len(sentences))
    print("The size of sentences graph is: ", len(sentences_graph))
    print("The output file is " + output_pairs_sentences_dir)
    print("Finished.")

    # Save last result
    result = []
    # Save dict in the form of {entity id: sentence id}
    # Save entity_pair in the list
    # Save entity in the list
    entity_sentence_dict = {}
    entity_pair = []
    entity = []

    for pairs in tqdm(pairs_contexts):
        hype_str = ''.join(word for word in pairs[0])
        hypo_str = ''.join(word for word in pairs[1])
        if hype_str not in entity:
            entity.append(hype_str)
        if hypo_str not in entity:
            entity.append(hypo_str)
        hype_num = entity_dict[hype_str] if hype_str in entity_dict else -1
        hypo_num = entity_dict[hypo_str] if hypo_str in entity_dict else -1
        hype_sentence, hypo_sentence = [], []

        if hype_num == -1:
            hype_sentence.append(hype_str)
        if hypo_num == -1:
            hypo_sentence.append(hypo_str)

        if hype_num != -1:
            if hype_num in entity_sentence_dict:
                hype_sentence.append(''.join(word for word in sentences[entity_sentence_dict[hype_num]]))
            else:

                for sen_index, line in enumerate(sentences_graph):
                    if line[0] == 0:
                        continue
                    elif hype_num in line and sen_index not in entity_sentence_dict.values():
                        hype_sentence.append(''.join(word for word in sentences[sen_index]))
                        entity_sentence_dict[hype_num] = sen_index
                        break
                if not hype_sentence:
                    hype_sentence.append(hype_str)

        if hypo_num != -1:
            if hypo_num in entity_sentence_dict:
                hypo_sentence.append(''.join(word for word in sentences[entity_sentence_dict[hypo_num]]))
            else:

                for sen_index, line in enumerate(sentences_graph):
                    if line[0] == 0:
                        continue
                    elif hypo_num in line and sen_index not in entity_sentence_dict.values():
                        hypo_sentence.append(''.join(word for word in sentences[sen_index]))
                        entity_sentence_dict[hypo_num] = sen_index
                        break
                if not hypo_sentence:
                    hypo_sentence.append(hypo_str)

        if hype_str+"_"+hypo_str not in entity_pair:
            entity_pair.append(hype_str+"_"+hypo_str)
        result.append([pairs[2]] + hype_sentence + hypo_sentence)
        # print(pairs[0], pairs[1])
        # print(pairs[2], hype_sentence, hypo_sentence)

    postive_count, negative_count = 0, 0

    for index, line in enumerate(result):
        if line[0] == 0:
            negative_count += 1
        elif line[0] == 1:
            postive_count += 1
    increment_count = postive_count - negative_count

    while increment_count > 0:
        entity_1 = random.choice(entity)
        entity_2 = random.choice(entity)
        if entity_1+"_"+entity_2 not in entity_pair:
            curr = []
            increment_count -= 1
            curr.append(0)
            if entity_1 in entity_dict and entity_dict[entity_1] in entity_sentence_dict:
                curr.append(''.join(word for word in
                                    sentences[entity_sentence_dict[entity_dict[entity_1]]]))
            else:
                curr.append(entity_1)
            if entity_2 in entity_dict and entity_dict[entity_2] in entity_sentence_dict:
                curr.append(''.join(word for word in
                                    sentences[entity_sentence_dict[entity_dict[entity_2]]]))
            else:
                curr.append(entity_2)
            result.append(curr)

    # With shuffle
    random.shuffle(result)
    test_dir = os.path.join(output_pairs_sentences_dir, "test.tsv")
    dev_dir = os.path.join(output_pairs_sentences_dir, "dev.tsv")
    train_dir = os.path.join(output_pairs_sentences_dir, "train.tsv")
    columns = ["label", "text_a", "text_b"]
    pd.DataFrame(result[:int(0.6*len(result))], columns=columns)\
        .to_csv(train_dir, sep='\t', index=False)
    pd.DataFrame(result[int(0.6*len(result)): int(0.8*len(result))], columns=columns)\
        .to_csv(dev_dir, sep='\t', index=False)
    pd.DataFrame(result[int(0.8*len(result)):], columns=columns)\
        .to_csv(test_dir, sep='\t', index=False)


if __name__ == '__main__':
    main()



