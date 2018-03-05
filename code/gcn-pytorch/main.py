import argparse
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np 
from torch.autograd import Variable
from utils import *

# define train settings
parser = argparse.ArgumentParser(description='PyTorch GCN')
parser.add_argument('--epochs', type=int, default=300, metavar='N')
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
    def __init__(self, input_dim, output_dim, dropout=0, is_bias=False,
                featureless=False, order_num=0):
        super(GraphicalConv2d, self).__init__()
        self.dropout = dropout
        self.is_bias = is_bias
        self.featureless = featureless
        self.order_num = order_num

        if self.is_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.constant(self.bias.data, 0)

        self.weights = nn.Parameter(torch.Tensor(input_dim, output_dim, order_num + 1))
        nn.init.xavier_normal(self.weights.data)

    def forward(self, x, adj):
        if not self.featureless:
            output = x.mm(self.weights[:, :, 0])
            for i in range(self.order_num):
                output.add_(adj.pow(i+1).mm(x).mm(self.weights[:, :, i+1]))
        else:
            pass
        
        if self.is_bias:
            output.add_(self.bias)

        return output.clamp(min=0)


# define new network structure
class GCNNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1=16, order_num=1):
        super(GCNNet, self).__init__()
        self.gcn1 = GraphicalConv2d(input_dim, hidden1, dropout=True, is_bias=True, 
                                    featureless=False, order_num=order_num)
        self.gcn2 = GraphicalConv2d(hidden1, output_dim, dropout=True, is_bias=True, 
                                    featureless=False, order_num=order_num)
        self.softmax = nn.Softmax()

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        x = self.gcn2(x, adj)
        x = self.softmax(x)
        return x

# load train/test data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')
adj = np.array(adj.toarray(), dtype=np.double)
print(adj)

input_dim = features.shape[1]
output_dim = y_train.shape[1]
num_sample = y_train.shape[0]

model = GCNNet(input_dim, output_dim, hidden1=8, order_num=2)

# define optimizer
# define loss + regularization
criterion = torch.nn.MSELoss(size_average=True)
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

# define train()
dtype = torch.FloatTensor
features = Variable(torch.from_numpy(features.toarray()).type(dtype), requires_grad=False)
adj = Variable(torch.from_numpy(adj).type(dtype), requires_grad=False)
y_train = Variable(torch.from_numpy(y_train).type(dtype), requires_grad=False)

for t in range(args.epochs):
#for t in range(10):
    output = model(features, adj)
    print(output)
    # masked cross_entropy loss
    loss = torch.sum(-y_train * torch.log(output+1e-9)) / torch.sum(y_train)
    print(t, loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# todo: define test()
y_test = Variable(torch.from_numpy(y_test).type(dtype), requires_grad=False)
output = model(features, adj)
_, predict = torch.max(output.data, 1)
_, labels = torch.max(y_test.data, 1)

masked_acc = (test_mask * (predict == labels)).sum() / test_mask.sum()

print('Evaluate Accuracy=', masked_acc)
