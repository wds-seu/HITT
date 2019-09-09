# config.py


class Config(object):
    embed_size = 300
    num_channels = 100
    kernel_size = [3,4,5]
    output_size = 2
    max_epochs = 20
    lr = 0.01
    batch_size = 64
    max_sen_len = 10
    dropout_keep = 0.8
    context_flag = 0