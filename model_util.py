import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

from util.data_util import get_Data, get_DataLoader

class CustomTrain:
    def __init__(self, gpu=False, dtype=torch.float32):
        # Set train info.

        if gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.dtype = dtype
        self.data_train_val = None
        self.data_test = None
        self.loader_train = None
        self.loader_val = None
        self.loader_test = None

        self.model_pool = {}

        print('- Train info -')
        print('Using device:', self.device)
        print('Using dtype:', self.dtype)
        print()


    def load_dataset(self, dset, transform=None, val_ratio=0.1, batch_size=64):
        # Load dataset & transform it.

        self.data_train_val, self.data_test = get_Data(dset, transform)
        self.loader_train, self.loader_val, self.loader_test = \
            get_DataLoader(self.data_train_val, self.data_test, val_ratio, batch_size)


    def load_data(self, data_train_val, data_test, val_ratio=0.1, batch_size=64):
        # Load data not dataset.

        self.data_train_val, self.data_test = data_train_val, data_test
        self.loader_train, self.loader_val, self.loader_test = \
            get_DataLoader(self.data_train_val, self.data_test, val_ratio, batch_size)


    def print_model_info(self, name):
        # Print model info
        
        summary(self.model_pool[name], self.data_train_val.data.shape[1:],
                device=self.device.type)


    def load_model(self, name, model, overlap=False):
        # Load own model to model_dict

        if name in self.model_pool and not overlap:
            assert False, 'same name in model pool'

        self.model_pool[name] = model.to(device=self.device, dtype=self.dtype)
        print('Model "%s" loaded. Summary below.' % name)
        self.print_model_info(name)


    def get_batch_accuracy(self, model, X, y, print_=False):
        # get accuracy on given batch data.
        
        model.eval()
        with torch.no_grad():
            X = X.to(device=self.device, dtype=self.dtype)
            y = y.to(device=self.device, dtype=torch.long)
            scores = model(X)
            _, preds = scores.max(1)

            num_correct = (y == preds).sum()
            num_samples = y.size(0)

            acc = float(num_correct) / num_samples
            if print_: print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, acc))
            return acc


    def get_loader_accuracy(self, model, loader, print_=False):
        # get accuracy on given data loader.

        if print_:
            if loader.dataset.train:
                print('Checking accuracy on validation set.')
            else:
                print('Checking accuracy on test set.')

        num_correct = 0
        num_samples = 0

        model.eval()
        with torch.no_grad():
            for X, y in loader:
                X = X.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=torch.long)
                scores = model(X)
                _, preds = scores.max(1)

                num_correct += (y == preds).sum()
                num_samples += y.size(0)

            acc = float(num_correct) / num_samples
            if print_: print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, acc))
            return acc


    def train(self, name, **kwargs):
        # Let's train the model.
        # kwargs: optim, criterion, lr, epochs, print_every

        assert name in self.model_pool, 'no such model in model pool'
        model = self.model_pool[name]

        optim = kwargs.get('optim', torch.optim.Adam)
        criterion = kwargs.get('criterion', F.cross_entropy)
        lr = kwargs.get('lr', 1e-3)
        epochs = kwargs.get('epochs', 1)
        print_every = kwargs.get('print_every', 50)
        
        optim = optim(model.parameters(), lr=lr)
        model = model.to(device=self.device)

        for epoch in range(epochs):
            print('epoch %d / %d' % (epoch+1, epochs))

            for t, (X, y) in enumerate(self.loader_train):
                model.train()
                X = X.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=torch.long)

                scores = model(X)
                loss = criterion(scores, y)

                optim.zero_grad()
                loss.backward()
                optim.step()

                if (t+1) % print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t+1, loss.item()))
                    self.get_loader_accuracy(model, self.loader_val, print_=True) 
                    print()

        self.model_pool[name] = model


    def test(self, name):
        # Let's test the model.

        assert name in self.model_pool, 'no such model in model pool'
        model = self.model_pool[name]
        self.get_loader_accuracy(model, self.loader_test, print_=True)














