# Reference to https://github.com/dbiir/UER-py

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    BertModel consists of three parts:
        - embedding: token embedding, position embedding, segment embedding
        - encoder: multi-layer transformer encoders
        - target: mlm ans nsp tasks

    """


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder,
    and target layers yield pre-trained models of different
    properties.
    :param args: config parameter
    :return:    model
    """
    embedding = globals()[args.embedding.capitalize() + "Embedding"](args, len(args.vocab))
    encoder = globals()[args.encoder.capitalize() + "Encoder"](args)
    model = Model(args, embedding, encoder)

    return model
