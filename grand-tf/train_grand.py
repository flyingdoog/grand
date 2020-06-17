from config import args,optimizer
import time
import tensorflow as tf
from utils import load_data,accuracy
from models import MLP, GCN
import numpy as np

dataset = args.dataset
A, features, onehot_labels, idx_train, idx_val, idx_test = load_data(dataset)
idx_unlabel = tf.range(len(idx_train), onehot_labels.shape[0]-1, dtype=tf.int32)

onehot_labels = onehot_labels.astype(np.float32)
clip_value_min = -2.0
clip_value_max = 2.0
labels = np.argmax(onehot_labels,axis=-1)

# build the model
model = MLP(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            input_droprate=args.input_droprate,
            hidden_droprate=args.hidden_droprate,
            use_bn = args.use_bn)


# prepare the tensors
features_tensor = tf.convert_to_tensor(np.array(features.todense()),dtype=tf.float32)
# labels_tensor = tf.convert_to_tensor(onehot_labels,dtype=tf.float32)
A_tensor = tf.SparseTensor(*A)

# random walk
def propagate(feature, A, order):
    x = feature
    y = feature
    for i in range(order):
        x = tf.sparse.sparse_dense_matmul(A, x)
        y += x
    return y/(order + 1.0)

# randomly drop edges
def rand_prop(features, training):
    n = features.shape[0]
    drop_rate = args.dropnode_rate
    if training:
        mask = tf.random.uniform((n,1),minval=0, maxval=1, dtype=tf.dtypes.float32)
        mask = tf.math.sign(mask-drop_rate)+1
        droped_mask = tf.clip_by_value(mask,clip_value_min=0, clip_value_max=1)
        features = tf.multiply(features ,droped_mask)

    else:
        features = features * drop_rate
    features = propagate(features, A_tensor, args.order)
    return features

# consistency loss
def consis_loss(logps, temp=args.tem):
    ps = [tf.math.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p / len(ps)

    sharp_p = (tf.math.pow(avg_p, 1. / temp) / tf.reduce_sum(tf.math.pow(avg_p, 1. / temp), axis=1, keepdims=True))
    loss = 0.
    for p in ps:
        p_sharp_p2 = tf.math.pow(p - sharp_p,2)
        loss += tf.reduce_mean(tf.reduce_sum(p_sharp_p2,axis=1))
    loss = loss / len(ps)
    return args.lam * loss


def train(epoch):
    t = time.time()
    X = features_tensor
    X_list = []
    K = args.sample

    with tf.GradientTape() as tape:
        for k in range(K):
            X_list.append(rand_prop(X, training=True))

        output_list = []
        for k in range(K):
            output_list.append(model(X_list[k],training=True))

        loss_train = 0.
        for k in range(K):
            y_true = onehot_labels[idx_train]
            y_pred = tf.gather(output_list[k],idx_train)
            loss_train += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
        loss_train = loss_train/K

        log_softmax_output_list = [tf.nn.log_softmax(out,-1) for out in output_list]
        loss_consis = consis_loss(log_softmax_output_list)

        l2_loss = args.weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        loss_train = loss_train + loss_consis+l2_loss
        grads = tape.gradient(loss_train, model.trainable_variables)
        # cliped_grads = [tf.clip_by_value(t, clip_value_min, clip_value_max) for t in grads]
    # optimizer.apply_gradients(zip(cliped_grads, model.trainable_variables))
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    y_pred = tf.gather(output_list[0],idx_train).numpy()
    acc_train = accuracy(y_pred=y_pred,y_true=labels[idx_train])

    if not args.fastmode:
        X = rand_prop(X, training=False)
        output = model(X,training=False)

    loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels[idx_val],logits=tf.gather(output,idx_val)))
    acc_val = accuracy(tf.gather(output,idx_val).numpy(),labels[idx_val])

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.numpy()),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_val: {:.4f}'.format(loss_val.numpy()),
          'acc_val: {:.4f}'.format(acc_val),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val, acc_val


def Train():
    # Train model
    t_total = time.time()
    loss_values = []
    acc_values = []
    bad_counter = 0
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    best_epoch = 0

    for epoch in range(args.epochs):
        l, a = train(epoch)
        loss_values.append(l)
        acc_values.append(a)

        print(bad_counter)

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:# or epoch < 400:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                model.save_weights(args.save_path + args.dataset)
            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        # print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_weights(args.save_path + args.dataset)



def test():
    X = features_tensor
    X = rand_prop(X, training=False)
    output = model(X,training=False)

    loss_test = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels[idx_test],logits=tf.gather(output, idx_test)))
    output = tf.nn.log_softmax(output,-1)
    acc_test = accuracy(tf.gather(output,idx_test).numpy(),labels[idx_test])

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.numpy()),
          "accuracy= {:.4f}".format(acc_test))

Train()
test()
