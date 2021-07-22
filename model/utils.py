import torch as t
import numpy as np
from torch_geometric.data import Dataset, Data
import scipy.io as sio
import os

device = t.device('cuda' if t.cuda.is_available() else 'cpu')


@t.no_grad()
def acc(net, labels, label_num, dataloader):
    # Calculate the confusion matrix and accuracy of the current batch
    confusion_matrix = t.zeros(3, 3).to(device)
    for i, data in enumerate(dataloader):
        data = data.to(device)
        pred = net(data).argmax(dim=1)
        truth = data.Y.view(-1, labels)[:, label_num]
        confusion_matrix[0, 0] += t.sum((pred == 0).long() * (truth == 0).long())
        confusion_matrix[0, 1] += t.sum((pred == 1).long() * (truth == 0).long())
        confusion_matrix[0, 2] += t.sum((pred == 2).long() * (truth == 0).long())
        confusion_matrix[1, 0] += t.sum((pred == 0).long() * (truth == 1).long())
        confusion_matrix[1, 1] += t.sum((pred == 1).long() * (truth == 1).long())
        confusion_matrix[1, 2] += t.sum((pred == 2).long() * (truth == 1).long())
        confusion_matrix[2, 0] += t.sum((pred == 0).long() * (truth == 2).long())
        confusion_matrix[2, 1] += t.sum((pred == 1).long() * (truth == 2).long())
        confusion_matrix[2, 2] += t.sum((pred == 2).long() * (truth == 2).long())
    return t.div(t.trace(confusion_matrix), t.sum(confusion_matrix)), confusion_matrix


def accuracy(mat):
    return t.trace(mat) / t.sum(mat)


def recall(mat):
    return mat[0][0] / t.sum(mat[0])


def precision(mat):
    return mat[0][0] / t.sum(mat[:, 0])


def F1_score(mat):
    return 2 * recall(mat) * precision(mat) / (recall(mat) + precision(mat))
