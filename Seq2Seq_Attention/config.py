# config.py

class Config(object):
    embed_size = 300
    hidden_layers = 1
    hidden_size = 64
    bidirectional = True
    output_size = 2
    max_epochs = 30
    lr = 0.1
    batch_size = 64
    dropout_keep = 0.8
    max_sen_len = 10 # Sequence length for RNN
    context_flag = 0