import torch
import torch.nn as nn


class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSigmoid(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, src, label, mask):
        """
        :param src:     [batch_size x seq_length]
        :param label:   [batch_size]
        :param mask:    [batch_size x seq_length]
        :return:        loss, logits
        """
        # Embedding.
        emb = self.embedding(src, mask)
        # Encoder.
        output = self.encoder(emb, mask)
        # Target
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))

        return loss, logits

