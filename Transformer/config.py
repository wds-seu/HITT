# config.py

class Config(object):
    embed_size = 300
    N = 1   # 6 in Transformer Paper
    d_model = 256   # 512 in Transformer Paper
    d_ff = 512  # 2048 in Transformer Paper
    h = 8
    dropout = 0.1
    output_size = 2
    lr = 0.00001
    max_epochs = 200
    batch_size = 128
    max_sen_len = 40
    context_flag = 2