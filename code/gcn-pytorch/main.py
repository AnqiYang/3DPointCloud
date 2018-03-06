import argparse
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np 
from torch.autograd import Variable
from utils import *

# define train settings
parser = argparse.ArgumentParser(description='PyTorch GCN')
parser.add_argument('--epochs', type=int, default=200, metavar='N')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W')
parser.add_argument('--hidden1', type=int, default=16, metavar='H',
                    help='Number of units in hidden layer 1.')
parser.add_argument('--dropout', type=float, default=0.8, metavar='N',
                    help='Dropout rate (1 - keep probability)')
parser.add_argument('--order-num', type=int, default=2, metavar='N',
                    help='Order num of Adjacency matrix')
parser.add_argument('--early-stopping', type=int, default=300, metavar='E',
                    help='Tolerance for early stopping.')
args = parser.parse_args()


# define model and gcn layer
# todo: define GraphicalConv
class GraphicalConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, act, dropout=0., is_bias=False,
                featureless=False, order_num=0):
        super(GraphicalConv2d, self).__init__()
        self.dropout = dropout
        self.is_bias = is_bias
        self.featureless = featureless
        self.order_num = order_num
        self.dropout = nn.Dropout(p=dropout)
        self.act = act

        if self.is_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.constant(self.bias.data, 0)

        self.weights = nn.Parameter(torch.Tensor(input_dim, output_dim, order_num + 1))
        #np.random.seed(10)
        nn.init.xavier_normal(self.weights.data)
        #for i in range(self.order_num + 1):
        #    self.weights[:, :, i].data = torch.from_numpy(np.random.rand(input_dim, output_dim))

    def forward(self, x, adj):
        x = self.dropout(x)
        if not self.featureless:
            output = x.mm(self.weights[:, :, 0])
            for i in range(self.order_num):
                output.add_(adj.pow(i+1).mm(x).mm(self.weights[:, :, i+1]))
        else:
            pass
        
        if self.is_bias:
            output.add_(self.bias)

        return self.act(output)


# define new network structure
class GCNNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1=8, dropout=0., order_num=1):
        super(GCNNet, self).__init__()
        self.gcn1 = GraphicalConv2d(input_dim, hidden1, act=F.relu, dropout=dropout, is_bias=True, 
                                    featureless=False, order_num=order_num)
        self.gcn2 = GraphicalConv2d(hidden1, output_dim, act=lambda x: x, dropout=dropout, is_bias=True, 
                                    featureless=False, order_num=order_num)
#        self.softmax = nn.Softmax()

    def forward(self, x, adj, mask):
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
#        x = self.softmax(x)
        x = mask.mm(x)
        return x

# load train/test data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

features = preprocess_features(features)
features = np.array(features.toarray(), dtype=np.float)
adj = np.array(adj.toarray(), dtype=np.float)

input_dim = features.shape[1]
output_dim = y_train.shape[1]
num_sample = y_train.shape[0]


for dropout in np.arange(10)/10:

    model = GCNNet(input_dim, output_dim, hidden1=8, dropout=dropout, order_num=2)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    if dropout == 0:
        mask = np.diag(train_mask)[0:np.sum(train_mask), :].astype(np.float32)
        y_train = np.where(np.dot(mask, y_train.data))[1]

        # define train()
        dtype = torch.FloatTensor
        features = Variable(torch.from_numpy(features).type(dtype), requires_grad=False)
        y_train = Variable(torch.from_numpy(y_train).type(torch.LongTensor), requires_grad=False)
        adj = Variable(torch.from_numpy(adj).type(dtype), requires_grad=False)
        mask = Variable(torch.from_numpy(mask).type(dtype), requires_grad=False)

    num_train = len(y_train)

    #for t in range(args.epochs):
    for t in range(150):
        output = model(features, adj, mask)
        # masked cross_entropy loss
        # loss = torch.sum(-y_train * torch.log(output+1e-9)) / torch.sum(y_train)
        loss = criterion(output, y_train)
        _, predict = torch.max(output.data, 1)
    
    
        acc = (predict == y_train.data).sum() / num_train
        print(t, loss.data[0], acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # todo: define test()
    if dropout == 0.:
        mask_test = np.diag(test_mask)[-np.sum(test_mask):, :].astype(np.float32)
        y_test = np.where(np.dot(mask_test, y_test))[1]

        mask_test = Variable(torch.from_numpy(mask_test).type(dtype))
        y_test = Variable(torch.from_numpy(y_test).type(torch.LongTensor), requires_grad=False)

    num_test = len(y_test)

    output = model(features, adj, mask_test)
    _, predict = torch.max(output.data, 1)

    masked_acc = (predict == y_test.data).sum() / num_test

    print('dropout=%.1f, Evaluate Accuracy=%.8f' % (dropout, masked_acc))
