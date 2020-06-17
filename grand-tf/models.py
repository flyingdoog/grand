import tensorflow as tf
from layers import GraphConvolution, MLPLayer

class MLP(tf.keras.Model):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn=False):
        super(MLP, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.is_cuda = is_cuda
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.use_bn = use_bn

    def call(self, x, training=True):

        if self.use_bn:
            x = self.bn1(x,training=training)

        if training:
            x = tf.nn.dropout(x, self.input_droprate)
        x = tf.nn.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x,training=training)
        if training:
            x = tf.nn.dropout(x, self.hidden_droprate)
        x = self.layer2(x)

        return x


class GCN(tf.keras.Model):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.use_bn = use_bn


    def call(self, x, adj,training=True):

        if self.use_bn:
            x = self.bn1(x,training=training)
        if training:
            x = tf.dropout(x, self.input_droprate)

        x = tf.nn.relu(self.gc1(x, adj))
        if self.use_bn:
            x = self.bn2(x,training=training)
        if training:
            x = tf.nn.dropout(x, self.hidden_droprate)
        x = self.gc2(x, adj)

        return x

