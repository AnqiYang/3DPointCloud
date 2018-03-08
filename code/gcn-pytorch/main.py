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

torch.set_printoptions(precision=8)

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
        self.dropout = nn.Dropout(p=dropout) #todo: figure out training / testing
        self.act = act

        if self.is_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.constant(self.bias.data, 0.)

        self.weights = nn.Parameter(torch.Tensor(input_dim, output_dim, order_num + 1))
        #nn.init.constant(self.weights.data, 1.)
        #self.weights.data = torch.from_numpy(np.ones((input_dim, output_dim, order_num+1), dtype=np.float32))
        nn.init.xavier_normal(self.weights.data)
        #np.random.seed(10)
        #for i in range(self.order_num + 1):
        #    self.weights[:, :, i].data = torch.from_numpy(np.random.rand(input_dim, output_dim))

    def forward(self, x, adj):
        #todo:
        #x = self.dropout(x)
        if not self.featureless:
            output = x.mm(self.weights[:, :, 0])
        for i in range(self.order_num):
        #for i in range(1):
                #output.add_(adj.pow(i+1).mm(x).mm(self.weights[:, :, i+1]))
                pre_sup = x.mm(self.weights[:, :, i+1])
                for j in range(i+1):
                    pre_sup = adj.mm(pre_sup)
                output.add_(pre_sup)
        else:
            pass
        if self.is_bias:
            #todo
            #pass
            output.add_(self.bias)
        
        return self.act(output)
        #return output

# define new network structure
class GCNNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1=8, dropout=0., order_num=1):
        super(GCNNet, self).__init__()
        self.gcn1 = GraphicalConv2d(input_dim, hidden1, act=F.relu, dropout=dropout, is_bias=True, 
                                    featureless=False, order_num=order_num)
        self.gcn2 = GraphicalConv2d(hidden1, output_dim, act=lambda x: x, dropout=dropout, is_bias=True, 
                                    featureless=False, order_num=order_num)
        self.softmax = nn.Softmax()

    def forward(self, x, adj):
        x = self.gcn1(x, adj)
        #print('     - gcn1 output:', x.data)
        x = self.gcn2(x, adj)
        #print('     - gcn2 output:', x.data)
        x = self.softmax(x)
#        x = mask.mm(x)
        return x


average_acc = 0
for i in np.arange(100):
    # load train/test data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

    # todo
    #features = np.loadtxt('features.txt', delimiter=',')
    #adj = np.loadtxt('adj.txt', delimiter=',')

    #print(features)
    #print(adj.sum())
    features = preprocess_features(features)
    features = np.array(features.toarray(), dtype=np.float)
    adj = preprocess_adj2(adj)
    #print(adj)
    #exit(0)
    adj = np.array(adj.toarray(), dtype=np.float)

    input_dim = features.shape[1]
    output_dim = y_train.shape[1]
    num_sample = y_train.shape[0]

    #features = np.random.randint(2, size=(num_sample, input_dim))
    #adj = np.random.randint(2, size=(num_sample, num_sample))

    #np.savetxt('features.txt', features, delimiter=',')
    #np.savetxt('adj.txt', adj, delimiter=',')

    #exit(0)

    model = GCNNet(input_dim, output_dim, hidden1=8, dropout=0.4, order_num=2)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    #criterion = nn.CrossEntropyLoss()

    #if i == 0:
    #mask = np.diag(train_mask)[0:np.sum(train_mask), :].astype(np.float32)
    #y_train = np.where(np.dot(mask, y_train.data))[1]

        # define train()
    dtype = torch.FloatTensor
        #features = Variable(torch.from_numpy(features).type(dtype), requires_grad=False)
        #y_train = Variable(torch.from_numpy(y_train).type(torch.LongTensor), requires_grad=False)
        #adj = Variable(torch.from_numpy(adj).type(dtype), requires_grad=False)
        #mask = Variable(torch.from_numpy(mask).type(dtype), requires_grad=False)
        
    features = Variable(torch.from_numpy(features).type(dtype), requires_grad=False)
    adj = Variable(torch.from_numpy(adj).type(dtype), requires_grad=False)
    y_train = Variable(torch.from_numpy(y_train).type(dtype), requires_grad=False)
    y_test = Variable(torch.from_numpy(y_test).type(dtype), requires_grad=False)
    train_mask = train_mask.astype(np.float32)
    test_mask = test_mask.astype(np.float32)
    train_mask = Variable(torch.from_numpy(train_mask).type(dtype), requires_grad=False)
    test_mask = Variable(torch.from_numpy(test_mask).type(dtype), requires_grad=False)
    
    num_train = torch.sum(train_mask)
    num_test = torch.sum(test_mask)

    # train
    for t in range(args.epochs):
    #for t in range(120):
        output = model(features, adj)
        # masked cross_entropy loss
        # loss = torch.sum(-y_train * torch.log(output+1e-9)) / torch.sum(y_train)
        loss = torch.sum(-torch.trace(y_train.mm(torch.log(output.t()+1e-32)))) / torch.sum(y_train)
        #loss = criterion(output, y_train)
        predict = torch.max(output.data, 1, keepdim=True)[1].type(dtype)
        labels = torch.max(y_train.data, 1, keepdim=True)[1].type(dtype)    

        acc = ((predict == labels).float().mul_(train_mask.data)).sum() / num_train
        print(t, loss.data[0], acc.data[0])
        # print(t, loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # test
    #dropout_fn = nn.Dropout(p=0.4)
    #features = dropout_fn(features)
    output = model(features, adj)
    predict = torch.max(output.data, 1, keepdim=True)[1].type(dtype)
    labels = torch.max(y_test.data, 1, keepdim=True)[1].type(dtype)

    test_acc = (predict == labels).float().mul_(test_mask.data).sum() / num_test
    
    dropout = 0.4
    print('dropout=%.1f, Evaluate Accuracy=%.8f' % (dropout, test_acc.data[0]))
    average_acc += test_acc.data[0]

print('Average accuracy on 100 run: %.8f' % (average_acc/100))
