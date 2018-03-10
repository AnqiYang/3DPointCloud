from __future__ import division
from __future__ import print_function

from multiprocessing import Process
import time
import tensorflow as tf

from utils import *
from models import GCN, MLP, GCNOur

# Set random seed
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_our', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')     #shz 200
# flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
# flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
# flags.DEFINE_float('order_num', 0, 'order number.')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 300, 'Tolerance for early stopping (# of epochs).')  #shz 200
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

def main(order_num, dropout, hidden_num):
    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    elif FLAGS.model == 'gcn_our':
        support = [preprocess_adj2(adj)]  # Not used
        num_supports = 1
        model_func = GCNOur
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    
    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    
    # Create model
    model = model_func(placeholders, input_dim=features[2][1], order_num=order_num, hidden1=hidden_num, logging=True)
    
    # Initialize session
    sess = tf.Session()
    
    
    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)
    
    
    # Init variables
    sess.run(tf.global_variables_initializer())
    
    cost_val = []
    
    # Train model
    for epoch in range(FLAGS.epochs):
    
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: dropout})
    
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        print(epoch)
        print(outs)   
        # Validation
        cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
        cost_val.append(cost)
        
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print('!!!!final epoch=', epoch)
            break
    
    # Testing
    test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)

    return acc, test_acc

def recursive_swap(swap_list, call_func, results, swap_index, input_index_list):
    if swap_index == len(swap_list):
        input_list = [s[i] for s, i in zip(swap_list, input_index_list)]
        results[tuple(input_index_list)], cross_val_acc = call_func(*input_list)
        print(input_list, cross_val_acc, results[tuple(input_index_list)])
        return

    for i in range(len(swap_list[swap_index])):
        input_index_list.append(i)
        recursive_swap(swap_list, call_func, results, swap_index + 1, input_index_list)
        input_index_list.pop()

def run(seed):
    print('seed: ', seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    swap_list = list()
    order_num = [2]   # revise byshz [1,2,3,4]
    swap_list.append(order_num)
    #dropout = [i/10 for i in range(10)]
    #todo
    dropout = [0.]
    swap_list.append(dropout)
    hidden_num = [8]
    swap_list.append(hidden_num)

    dimention = [len(h) for h in swap_list]
    test_result = np.ones(dimention)
    input_index_list = list()
    recursive_swap(swap_list, main, test_result, 0, input_index_list)


if __name__ == "__main__":
    for i in range(100):
        seed = i + 100
        p = Process(target=run, args=(seed, ))
        p.start()
        p.join()
