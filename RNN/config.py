# config.py

class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 32
    bidirectional = True
    output_size = 2
    max_epochs = 20
    lr = 0.1
    batch_size = 64
    max_sen_len = 10 # Sequence length for RNN
    dropout_keep = 0.8
    context_flag = 0