import torch
import torch.nn as nn
import numpy as np

def softmax(x) :
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def cross_entropy(actual, predicted) :
    loss = -np.sum(actual * np.log(predicted))
    return loss

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])

Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(l1)
print(l2)

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print(predictions1)
print(predictions1)