import tensorflow as tf


class MLPLayer(tf.keras.Model):
    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        initializer = tf.initializers.glorot_normal()
        self.weight = self.add_weight('weight', [in_features, out_features],dtype=tf.float32,  initializer=initializer)
        if bias:
            self.bias = self.add_weight('bias', [out_features], dtype=tf.float32, initializer="zero")
        else:
            self.bias = None
    def call(self, input):
        output = tf.matmul(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GraphConvolution(MLPLayer):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__(in_features,out_features,bias)

    def call(self, input, adj):
        support = tf.mm(input, self.weight)
        output = tf.sparse.sparse_dense_matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
