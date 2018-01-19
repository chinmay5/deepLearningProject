from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def check_accuracy(model, loader):
        num_correct = 0
        num_samples = 0
        model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
	for x, y in loader:
	     x_var = Variable(x)
             y_var = Variable(y)
	     scores = model(x_var)
             loss = self.loss_func(scores, y_var)
             self.val_loss_history.append(loss)     
             _, preds = scores.data.cpu().max(1)
	     num_correct += (preds == y).sum()
	     num_samples += preds.size(0)
	acc = float(num_correct) / num_samples
        self.val_acc_history.append(acc)
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.
        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        params = [param for param in model.parameters() if param.requires_grad]

        optim = self.optim(params, **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        
        #The logic implementation
        #We would be storing the value of outputs in each epoch
        #Storing the value of error and accuracy for visualisation
        for epoch in range(num_epochs):
            num_correct = 0
            num_samples = 0
            print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
            model.train()
            for t, (x, y) in enumerate(train_loader):

                x_var = Variable(x)
                y_var = Variable(y)

                scores = model(x_var)
                loss = self.loss_func(scores, y_var)

                if (t + 1) % 100 == 0:
                    print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

                model.eval()
                #Appending the loss history
                self.train_loss_history.append(loss)
                _, preds = scores.data.cpu().max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                model.train()
                
                optim.zero_grad()
                loss.backward()
                optim.step()

            acc = float(num_correct) / num_samples
            self.train_acc_history.append(acc)
            print('Training acc for epoch %d is %.3f' %(epoch+1,acc))
            check_accuracy(model, val_loader)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
