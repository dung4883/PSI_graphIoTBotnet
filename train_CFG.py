from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy
import pandas
import csv
import random
import pickle
from model import TextCNN

from utils import utils
import matplotlib.pyplot as plt

def getClass(item):
    item = item[25:]
    item = item[:item.find('.')]
    if (int(item) < 2838):
        return 1
    else:
        return 0

def prepareData(file):
    a = utils()
    data = a.readFile(file)
    x = []
    y = []
    for item in data:
        #print(item)
        x.append(data[item])
        y.append(getClass(item))
    return x, y

def test(model, X, Y):
    model.eval()
    predLabel = []
    #Y = Y[:1872]
    for i in range(len(X)/52):
        batch_x = X[i * 52 : (i + 1) * 52]
        batch_x = Variable(torch.FloatTensor(batch_x)).cuda()
        pred = model(batch_x, len(batch_x))
        predLabel.extend(pred.data.max(1)[1].cpu().numpy())
    print(f1_score(Y, predLabel))
    print(Y)
    predLabel = np.array(predLabel)
    # Draw ROC, AUC
    fpr, tpr, thresholds = roc_curve(Y, predLabel.round(decimals=3), pos_label = 1)
    aucValue = auc(fpr, tpr)
    print(aucValue)
    plt.figure()
    lw =2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def train(x, y):
    model = TextCNN()
    model = model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(size_average=False)

    for epoch in range(50):
        total = 0
        for i in range(0, len(x)/64):
            batch_x = x[i*64:(i+1)*64]
            batch_y = y[i*64:(i+1)*64]
            batch_x = Variable(torch.FloatTensor(batch_x)).cuda()
            batch_y = Variable(torch.LongTensor(batch_y)).cuda()
            optimizer.zero_grad()
            model.train()
            pred = model(batch_x, 64)
            loss = criterion(pred, batch_y)
            #print(loss)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=3)
            total += np.sum(pred.data.max(1)[1].cpu().numpy() == batch_y.data.cpu().numpy())
            optimizer.step()
        print("epoch ", epoch + 1, " acc: ", float(total)/len(x))
    return model

if __name__ == '__main__':
    data_x, data_y = prepareData('/media/aisu/Others/Hoang/SOIS_2018/CNN_ATrung/data/PSI_dims_1024_epochs_100_lr_0.3_embeddings.txt')
    data_x, data_y = shuffle(data_x, data_y)
    data_x_train = data_x[:3968]
    data_y_train = data_y[:3968]
    data_x_test = data_x[3968:]
    data_y_test = data_y[3968:]
    print(len(data_x_test), len(data_y_test))
    #print(len(data_x_test))
    model = train(data_x, data_y)
    test(model, data_x_test, data_y_test)
    with open(r"model.pkl", "wb") as output_file:
        pickle.dump(model, output_file)