import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
from preprocess import *

"""
Class: Flatten
==============
Flattens a tensor of shape (N, F, C, H, W) to be of shape (N, F*C*H*W).
==============
"""
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)

"""
Class: Unflatten
==============
Unflattens a tensor of shape (N, C*H*W) to be of shape (N, C, H, W).
==============
"""
class Unflatten(nn.Module):
    def __init__(self, N=-1, C=128, H=7, W=7):
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

"""
Function: getOptimizer
======================
Returns an optimizer for the type specified by optimType.
======================
input:
    m: model to create optimizer for
    optimType: type of optimizer
        'adam', 'rmsprop', 'sgd'
    lr: learning rate
    alpha: alpha value for optimizer
    betas: beta1 and beta2 values for optimizer
    momentum: momentum for optimizer
output:
    optimizer: optimizer for the model m
"""
def getOptimizer(m, optimType='adam', lr=1e-3, alpha=0.9, betas=(0.5, 0.999), momentum=0.9):
    optimizer = None

    if (optimType == 'adam'):
        optimizer = optim.Adam(m.parameters(), lr=lr, betas=betas)
    elif (optimType == 'rmsprop'):
        optimizer = optim.RMSprop(m.parameters(), lr=lr, alpha=alpha, momentum=momentum)
    elif (optimType == 'sgd'):
        optimizer = optim.SGD(m.parameters(), lr=lr, momentum=momentum)
    else:
        print("Unsupported optimizer type")

    return optimizer

"""
Function: initializeWeights
===========================
Initializes the weights of the model using the xavier uniform method.
===========================
input:
    m: model
output:
    None
"""
def initializeWeights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)

"""
Function: defineModel
==================
Constructs a model for ASL classification.
==================
input:
    H: height of imgs
    W: width of imgs
    num_classes: number of classes to base scores off of
output:
    m: model created
"""
def defineModel(H, W, num_classes):

    m = nn.Sequential(
        nn.Conv2d(1, 32, 7, stride=1, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(11*41, num_classes)
    )

    m.apply(initializeWeights)

    return m

"""
Function: checkAccuracy
=======================
Checks the accuracy of the model m on the validation data data_val.
=======================
input:
    m: model whose accuracy we are checking
    data_val: validation set of the data
output:
    None
"""
def checkAccuracy(m, valX, valY):
    num_correct = 0
    num_samples = 0

    # Set model to evaluation mode
    m.eval()

    with torch.no_grad():
        for c in range(len(valX)):
            x, y = valX[c], valY[c]

            # Get scores
            scores = m(x)

            # Predictions of correct response
            _, preds = scores.max(1)

            # Determine the number of correct values
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

    return acc

"""
Function: train
===============
Trains the model m and saves it to path_to_model.
===============
input:
    m: model to be trained
    data_train: training set of the data
    data_val: validation set of the data
    path_to_model: directory where the model will be saved
    opt_params: tuple of parameters used in getOptimizer
        type: 'adam', 'rmsprop', 'sgd'
        lr: learning rate
        alpha: alpha value
        betas: tuple of beta1 and beta2
        momentum: momentum value
    num_epochs: number of epochs to run
    show_every: number to print statistics every show_every iteration
output:
    None
"""
def train(m, trainX, trainY, valX, valY, opt_params, model_name, path_to_model="../Models/",
          path_to_loss="../Plots/Loss/", path_to_acc="../Plots/Accuracy/", num_epochs=1, show_every=500):
    print("=====Training=====")

    iter_count = 0
    type, lr, alpha, betas, momentum = opt_params
    optimizer = getOptimizer(m, optimType=type, lr=lr, betas=betas, momentum=momentum)

    loss_array = np.zeros(data_train.shape[0] * num_epochs, dtype=np.float)
    acc_train = np.zeros(num_epochs, dtype=np.float)
    acc_val = np.zeros(num_epochs, dtype=np.float)

    for epoch in range(num_epochs):
        for c in range(len(trainX)):
            x, y = trainX[c], trainY[c]

            # Put model into training mode
            m.train()

            # Determine scores from model
            scores = m(x)

            # Determine loss
            loss = F.cross_entropy(scores, y)

            # Zero out all grads in the optimizer
            optimizer.zero_grad()

            # Perform backward pass from loss
            loss.backward()

            # Update parameters of the model
            optimizer.step()

            # Store loss in loss_array
            loss_array[iter_count] = loss.item()

            # Print the update of the loss
            if (iter_count % show_every == 0):
                print('Iteration %d, loss = %.4f' % (iter_count, loss.item()))
                checkAccuracy(m, valX, valY)
                print()

            iter_count += 1

        acc_train[epoch] = checkAccuracy(m, data_train)
        acc_val[epoch] = checkAccuracy(m, data_val)

    # # Save model
    # saveData(m, path_to_model + model_name)
    #
    # # Plot the loss
    # plotLoss(loss_array, path_to_loss + model_name + ".png")
    #
    # # Plot the accuracy
    # plotAccuracy(acc_train, acc_val, path_to_acc + model_name + ".png")

def model(mode, path_to_model="../Models/", opt_params=('adam', 1e-3, 0.9, (0.5, 0.999), 0.9)):
    if (mode == 'train'):
        trainX, trainY = gatherDataAsArray("train", "../Arrays/train")
        valX, valY = gatherDataAsArray("val", "../Arrays/val")

        trainX, trainY = reformData(trainX, trainY)
        valX, valY = reformData(valX, valY)

        _, num_classes = np.unique(trainY, return_counts=True)
        H, W = 656, 176

        model_name = "model_test"

        m = defineModel(H, W, num_classes)

        train(m, trainX, trainY, valX, valY, opt_params, model_name)
    elif (mode == 'test'):
        testX, testY = gatherDataAsArray("test", "../Arrays/test")
        testX, testY = reformData(testX, testY)
        m = pickle.load(open(path_to_model + model_name, 'rb'))

        checkAccuracy(m, testX, testY)

def splitData(path_to_vids):
    train_folder, val_folder, test_folder = "train", "val", "test"

    for path, subdirs, files in os.walk(path_to_vids):
        if (files[0] == '.DS_Store'):
            continue

        val, test, train = np.split(np.asarray(files), [1, 2])
        assert(len(train) > 0 and len(val) > 0 and len(test) > 0), "wrong dataSplit"

        for vid in train:
            copy(path + "/" + vid, train_folder + path[len(path_to_vids)::] + "/")

        for vid in val:
            copy(path + "/" + vid, val_folder + path[len(path_to_vids)::] + "/")

        for vid in test:
            copy(path + "/" + vid, test_folder + path[len(path_to_vids)::] + "/")

"""
Function: reformData
====================
Converts the tuple of numpy arrays to tuple of tensors.
====================
input:
    data: tuple of numpy arrays
output:
    newData: tuple of tensors
"""
def reformData(dataX, dataY):
    N, F, H, W = dataX.shape

    newDataX = torch.tensor(dataX)
    newDataX = newDataX.to(dtype=torch.float32)
    newDataY = torch.tensor(dataY, dtype=torch.long)

    return newDataX, newDataY
