# train.py

from utils import *
from model import *
from config import Config
import sys
import joblib
import torch.optim as optim
from torch import nn
import torch

if __name__=='__main__':
    config = Config()
    train_file = '../data/pairs_with_context.pkl'
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    embedding_file = '../data/chinese.w2c.300d.iter5'
    if len(sys.argv) > 3:
        embedding_file = sys.argv[2]
    
    # w2v_file = '/home/wds/Documents/gp/Text-Classification-Models-Pytorch-master/data/glove.840B.300d.txt'

    dataset = Dataset(config)
    # dataset.load_data(w2v_file, train_file, test_file)
    dataset.load_my_data(embedding_file, train_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = TextRNN(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    NLLLoss = nn.NLLLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
    
    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))