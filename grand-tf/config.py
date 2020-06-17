import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--save_path', type=str, default='./checkpoints/', help='dir to checkpoints')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--input_droprate', type=float, default=0.5,
                    help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden_droprate', type=float, default=0.5,
                    help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--dropnode_rate', type=float, default=0.5,
                    help='Dropnode rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--order', type=int, default=5, help='Propagation step')
parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument('--dataset', type=str, default='cora', help='Data set')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
#dataset = 'citeseer'
#dataset = 'pubmed'
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
import tensorflow as tf
seed = args.seed
random.seed(args.seed)
np.random.seed(seed)
if tf.__version__.startswith('2'):
    tf.random.set_seed(seed)
    optimizer = tf.keras.optimizers.Adam(args.lr)

else:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)
    tf.set_random_seed(seed)

    args.input_droprate = 1-args.input_droprate
    args.hidden_droprate = 1-args.hidden_droprate
    args.dropnode_rate = 1-args.dropnode_rate
    optimizer = tf.compat.v1.train.AdamOptimizer(args.lr)


