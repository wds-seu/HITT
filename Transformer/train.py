# train.py

from utils import *
from model import *
from config import Config
import sys
import numpy as np
import torch.optim as optim
import visdom
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
    
    dataset = Dataset(config)
    dataset.load_my_data(embedding_file, train_file)

    # show the loss tendency
    track_loss = 0
    global_step = 0
    vis = visdom.Visdom(env=u"train_loss")
    win = vis.line(X=np.array([global_step]), Y=np.array([track_loss]))
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = Transformer(config, len(dataset.vocab), dataset.word_embeddings)
    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
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
        vis.line(X=np.array([global_step]), Y=np.array([np.exp(train_loss[0])]), win=win, update='append')
        global_step += 1

    train_acc = evaluate_model(model, dataset.train_iterator)
    val_acc = evaluate_model(model, dataset.val_iterator)
    test_acc = evaluate_model(model, dataset.test_iterator)

    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))