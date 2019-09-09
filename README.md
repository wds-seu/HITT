# README

## HITT: Hypernymy Identification in Chinese Traffic Legal Text

### Requirements

* Python-3.7.3
* Pandas-0.24.2
* Numpy-1.15.4
* Pytorch-1.1.0
* Torchtext-0.4.0
* StanfordCoreNLP-3.9.1.1
* joblib-0.13.2

### Usage

1. Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/download.html) toolkit and the package for Chinese processing;
2. Download pre-trained [word embedding](https://github.com/Embedding/Chinese-Word-Vectors)
3. Use Stanford CoreNLP toolkit to parse the row sentence data and get the sequence of all the sentences by syntactic parsing;
4. Config the config.py file, use daojiao_pairs.xlsx file run the context_finder.py file to find the context of each pair in all sentence sequences;
5. Save the entity pairs, label information together with context information as input data;
6. Use Stanford CoreNLP to parse the input data into input sequence;
7. Run following command in all model files:

python train.py <path_to_training_file> <path_to_word_embedding_file>

