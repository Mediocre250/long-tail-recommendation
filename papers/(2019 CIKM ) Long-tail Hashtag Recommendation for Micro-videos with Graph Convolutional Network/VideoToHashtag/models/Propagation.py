import torch.nn as nn
from torch.nn import Parameter
from util import *

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNHashtag(nn.Module):
    def __init__(self, num_classes, adj_matrix, in_channel):
        super(GCNHashtag, self).__init__()

        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 150)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, adj_matrix)
        self.A = Parameter(torch.from_numpy(_adj).float())

    def forward(self, init_hashtag_matrix):

        adj = gen_adj(self.A).detach()
        x = self.gc1(init_hashtag_matrix, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        return x

class PropagationModel(nn.Module):
    def __init__(self, adj_file, hashtag_num, embed_size):
        super(PropagationModel, self).__init__()
        self.adj_file = adj_file
        self.hashtag_num = hashtag_num
        self.embed_size = embed_size
        self.init_hashtag_matrix = Parameter(torch.Tensor(self.hashtag_num, self.embed_size))
        self.adj_matrix = np.load(self.adj_file).astype(float)
        self.gcnhashtag = GCNHashtag(self.hashtag_num, self.adj_matrix, self.embed_size)

    def forward(self):
        hashtag_matrix = self.gcnhashtag(self.init_hashtag_matrix)
        return hashtag_matrix








