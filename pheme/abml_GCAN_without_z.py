# coding: utf-8
# In[ ]:

'''
python maml-GCAN.py --inner_lr=0.01 --meta_lr=0.005 --num_inner_updates=2 --k_shot=3 --num_val_shots=7  --num_epochs=10  --resume_epoch=0

'''
import numpy as np
from numpy import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch
import os
import random
import itertools
import sys
import csv
import argparse
import keras
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Embedding, Conv2D, Dense, Reshape, GRU, \
                         AveragePooling1D, Flatten, BatchNormalization
from keras.models import Model, load_model

from GCAN import preprocess_adj_tensor, MultiGraphCNN, coattention, cocnnattention
from dataset import DatasetBuilder
# from task_generator import load_raw_data, gen_topics, gen_raw_tasks, sample_task_from_raw_task, \
#     text_preprocess, gen_raw_tasks_by_label, sample_task_from_raw_task_by_label, load_struct_data, \
#     DATA_DIR, NO_BELOW, NO_ABOVE, MIN_COUNT
from utils import cos_similarity, f1_score, precision, recall, gen_topics_embedding, \
    load_raw_dataset, get_vocab_size, sample_task_from_raw_task_by_label, \
    load_struct_data, split_dataset, print_layer
from cfg import *

#using cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------------------------------------
# Setup input parser
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables for MAML.')

parser.add_argument('--datasource', type=str, default='twitter15', help='Datasource: twitter15, twitter16')
parser.add_argument('--n_topic', type=int, default=10, help='Number of topics in datasource')
parser.add_argument('--n_way', type=int, default=2, help='Number of classes per task')
parser.add_argument('--k_shot', type=int, default=3, help='Number of training samples per class or k-shot')
parser.add_argument('--topic_split_rate', type=list, default=[6,1,2], \
    help='Dataset split ratio')
parser.add_argument('--seed', type=int, default=10, help='Seed for splitting train-test set')
parser.add_argument('--num_val_shots', type=int, default=3, help='Number of validation samples per class')
parser.add_argument('--sample_rate', type=int, default=4, help='Rate for sampling tasks from topics')
parser.add_argument('--batch_size', type=int, default=5, help='Number of tasks per batch')
parser.add_argument('--batch_per_epoch', type=int, default=2, help='Number of batches per epoch')
parser.add_argument('--retweet_user_size', type=int, default=30, help='The length of retweet propagation you want to utilize')

parser.add_argument('--inner_lr', type=float, default=5e-3, help='Learning rate for task adaptation')
parser.add_argument('--num_inner_updates', type=int, default=10, help='Number of gradient updates for task adaptation')
parser.add_argument('--meta_lr', type=float, default=2e-3, help='Learning rate of meta-parameters')
parser.add_argument('--lr_decay', type=float, default=0.98, help='Decay factor of meta-learning rate (<=1), 1 = no decay')
# parser.add_argument('--minibatch', type=int, default=5, help='Number of tasks per minibatch to update meta-parameters')

parser.add_argument('--num_epochs', type=int, default=20, help='How many 10,000 tasks are used to train?')
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')


parser.add_argument('--log_file', type=str, default="log_non_z.txt", help='Name of log file')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

# parser.add_argument('--num_tasks_sampled', type=int, default=1, help='Number of tasks sampled from per topic')

args = parser.parse_args()

# --------------------------------------------------
# Parse dataset and related variables
# --------------------------------------------------
datasource = args.datasource
print('Dataset = {0:s}'.format(datasource))

num_topics = args.n_topic
print('Number of topics(tasks) = {}'.format(num_topics))

seed = args.seed
print('Random seed = {}'.format(seed))

topic_split_rate = args.topic_split_rate
print('Train-val-test = {}'.format(topic_split_rate))

train_flag = args.train_flag
print('Learning mode = {0}'.format(train_flag))

num_classes_per_task = args.n_way
print('Number of ways = {0:d}'.format(num_classes_per_task))

num_training_samples_per_class = args.k_shot
print('Number of shots = {0:d}'.format(num_training_samples_per_class))

num_val_samples_per_class = args.num_val_shots
print('Number of validation samples per class = {0:d}'.format(num_val_samples_per_class))

# num_tasks_sampled = args.num_tasks_sampled
# print('Number of tasks sampled from per topic = {0:d}'.format(num_tasks_sampled))

num_samples_per_class = num_training_samples_per_class + num_val_samples_per_class


# ----------------------------------------------
# Parse training parameters
# ----------------------------------------------
inner_lr = args.inner_lr
print('Inner learning rate = {0}'.format(inner_lr))

num_inner_updates = args.num_inner_updates
print('Number of inner updates = {0:d}'.format(num_inner_updates))

meta_lr = args.meta_lr
print('Meta learning rate = {0}'.format(meta_lr))

batch_size = args.batch_size
print('Batch size = {0}'.format(batch_size))

batch_per_epoch = args.batch_per_epoch
print("Number of batches per epoch:{}".format(batch_per_epoch))

retweet_user_size = args.retweet_user_size
print("The length of retweet propagation:{}".format(retweet_user_size))

# num_meta_updates_print = int(1000 / num_tasks_per_minibatch)

num_epochs_save = 1

num_epochs = args.num_epochs

# expected_total_tasks_per_epoch = 10
# num_tasks_per_epoch = int(expected_total_tasks_per_epoch / num_tasks_per_minibatch)*num_tasks_per_minibatch

# expected_tasks_save_loss = 10000
# num_tasks_save_loss = int(expected_tasks_save_loss / num_tasks_per_minibatch)*num_tasks_per_minibatch

# uncertainty_flag = args.uncertainty_flag

resume_epoch = args.resume_epoch

sample_rate = args.sample_rate

lr_decay = args.lr_decay
print('Learning rate decay = {0}'.format(lr_decay))

log_file = args.log_file
print("Log file: {}".format(log_file))


# #parameters setting
# SOURCE_TWEET_OUTPUT_DIM = 32
# OUTPUT_DIM = 64 
# GCN_FILTERS_NUM = 1
# GCN_OUTPUT_DIM = 32
# CNN_FILTER_SIZE = (3,3)
# CNN_OUTPUT_DIM = 32
# STRIDE_SIZE = 1
# CNN_OUTPUT_LENGTH = int(retweet_user_size*USER_FTS_DIM/STRIDE_SIZE)
# MIN_EPOCH_NUM = 50
# MAX_EPOCH_NUM = 100

if not os.path.exists("model/inner_model"):
    os.makedirs("model/inner_model")
INNER_MODEL_PATH = "model/inner_model/abml_without_z_{}retweeter.h5".format(retweet_user_size)

#evaluation
def f1_score(y_true, y_pred):
    #calculate tp、tn、fp、fn
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    
    
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    #计算f1
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return f1

def precision(y_true, y_pred):
    #calculate tp、tn、fp、fn
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    
   
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    #calculate f1
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return p

def recall(y_true, y_pred):
    #calculate tp、tn、fp、fn
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)
    
    
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    #calculate f1
    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return r

# def meta_train(train_tasks, val_tasks, net, theta, test_data):
def meta_train(train_ids, val_ids, net, theta, test_ids, dataset):
    
    # support_shots = num_training_samples_per_class * num_classes_per_task
    # query_shots = num_classes_per_task * num_classes_per_task
    support_shots = num_training_samples_per_class * sample_rate
    query_shots = num_classes_per_task * sample_rate

    for epoch in range(resume_epoch, resume_epoch + num_epochs):

        print("Meta epoch_{}:".format(epoch+1))
        # print("theta[0][0]:\n{}".format(theta[0][0]))
        task_num = len(train_ids)
        # initialize gradients with zero
        gradients = []
        for item in theta['mean']:
            gradients.append(np.zeros(item.shape))

        # batch_num = int(task_num/batch_size+0.5)
        batch_num = batch_per_epoch
        for batch in range(batch_num):
            np.random.seed()
            batch_idxs = np.random.choice(np.arange(task_num), size=batch_size, \
                replace=False, p=None)
            batch_ids = [train_ids[batch_idx] for batch_idx in batch_idxs]
            for task_ids in batch_ids:
                # run a task
                # load data of a task
                net.load_weights(INNER_MODEL_PATH)
                # task_s, task_q = get_support_query_data(train_task, num_training_samples_per_class)
                # print(task_ids)
                task_sample_ids = sample_task_from_raw_task_by_label(task_ids, support_shots, query_shots)

                task_s = load_struct_data([task_sample_ids['s']], dataset)[0]
                task_q = load_struct_data([task_sample_ids['q']], dataset)[0]

                # print(len(task_s))

                adapt_to_task(task_s, net, theta)
                # we valuate the model for every task
                WX_q, X_q, cnnX_q, MX_q, y_q = parse_task_data(task_q)
                # y_pred = net.predict([WX_q, X_q, cnnX_q, MX_q])
                weights_1 = net.get_weights().copy()
                net.fit([WX_q, X_q, cnnX_q, MX_q], y_q, epochs=1)
                weights_2 = net.get_weights().copy()
                
                # compute gradients
                for idx in range(len(weights_1)):
                    gradients[idx] += ((weights_1[idx] - weights_2[idx])/inner_lr)

            # update theta
            for idx in range(len(theta['mean'])):
                theta['mean'][idx] = theta['mean'][idx] - gradients[idx]*(meta_lr*pow(lr_decay, epoch))/batch_size
                theta['logSigma'][idx] = theta['logSigma'][idx] - gradients[idx]*(meta_lr*pow(lr_decay, epoch))/batch_size\
                    *np.exp(theta['logSigma'][idx])
        # evaluate the model
        
        metrics = validate_classification(val_ids, net, theta, dataset) 
        for score in metrics:
            print(score)
        val_sizes = task_sizes[val_idxs]
        val_w = val_sizes/sum(val_sizes)
        mean_metrics = np.array(metrics).T.dot(np.array(val_w))
        print("\nVal average: [loss, accuracy, f1_score, precision, recall]\n{}\n".format(\
                mean_metrics))
        with open(log_file, 'a+') as f:
            f.write("Val average epoch={}: {}\n".format(\
               epoch+1, mean_metrics))

        test_flag = True
        if test_flag:
            # evaluate the model
            metrics = validate_classification(test_ids, net, theta, dataset)
            for score in metrics:
                print(score)
            test_sizes = task_sizes[test_idxs]
            test_w = test_sizes/sum(test_sizes)
            mean_metrics = np.array(metrics).T.dot(np.array(test_w))
            print("\nTest average: [loss, accuracy, f1_score, precision, recall]\n{}\n".format(\
                    mean_metrics))
            with open(log_file, 'a+') as f:
                f.write("Test average epoch={}: {}\n".format(\
                   epoch+1, mean_metrics))

        # #-------------------------------------------
        # print("test_data:")
        # fixed_weights = net.get_weights().copy()    # advoid updating the meta_model and the inner_model
        # metrics = validate_classification(test_data, net, theta, train_flag=False, csv_flag=False)
        # net.set_weights(fixed_weights)  # advoid updating the meta_model and the inner_model
        # for score in metrics:
        #     print(score)
        # #-------------------------------------------

        if ((epoch + 1)% num_epochs_save == 0):
            checkpoint = {
                'theta': theta
            }
            print('SAVING WEIGHTS...')
            checkpoint_filename = 'Epoch_{0:d}.pt'.format(epoch + 1)
            print(checkpoint_filename)
            torch.save(checkpoint, os.path.join(dst_folder, checkpoint_filename))
        print()

def adapt_to_task(task, inner_model, theta):
    # load task data
    WX, X, cnnX, MX, y = parse_task_data(task)
    # print("WX shape:{}".format(WX.shape))
    # print("X shape:{}".format(X.shape))
    # print("cnnX shape:{}".format(cnnX.shape))
    # print("MX shape:{}".format(MX.shape))
    w0 = sample_nn_weight(theta)
    inner_model.set_weights(w0)
    # print(WX.shape, X.shape, cnnX.shape, MX.shape)
    inner_model.fit([WX, X, cnnX, MX], y, epochs=num_inner_updates, verbose=0)
    # y_pred = inner_model.predict([WX, X, cnnX, MX])


    # net_weights = inner_model.get_weights()
    # # load inner_model weights
    # w_task = []
    # for item in net_weights:
    #     w_task.append(item.copy)
    # return w_task
    return None

def sample_nn_weight(meta_params):
    # w = theta['mean']
    w = []
    for idx in range(len(meta_params['mean'])):
        np.random.seed()
        eps_sampled = np.random.random(meta_params['mean'][idx].shape)
        w.append(meta_params['mean'][idx] + eps_sampled * np.exp(meta_params['logSigma'][idx]))
    # print("mean:\n{}\nlogSigma:\n{}".format(meta_params['mean'][0][0], \
    #     np.exp(meta_params['logSigma'][0])[0]))
    return w

def predict_label_score(task, w):

    raw_scores = net.forward(x=x, w=w)
    sm_scores = sm(raw_scores)

    prob, y_pred = torch.max(input=sm_scores, dim=1)
    return y_pred, prob.detach().cpu().numpy()

def validate_classification(tasks_ids, net, theta, dataset, train_flag=False, csv_flag=False):
    support_shots = num_training_samples_per_class
    query_shots = num_classes_per_task
    # support_shots = num_training_samples_per_class * sample_rate
    # query_shots = num_classes_per_task * sample_rate
    metrics = []
    
    # total_val_samples_per_task = num_val_samples_per_class * num_classes_per_task
    for task_ids in tasks_ids:
        net.load_weights(INNER_MODEL_PATH)
        # net.optimizer.zero_grad()
        # task_s, task_q = get_support_query_data(task, num_training_samples_per_class)
        # task_sample_ids = sample_task_from_raw_task(task_ids, support_shots, query_shots)
        task_sample_ids = sample_task_from_raw_task_by_label(task_ids, support_shots, query_shots, False)
        task_s = load_struct_data([task_sample_ids['s']], dataset)[0]
        task_q = load_struct_data([task_sample_ids['q']], dataset)[0]
        # w_task = adapt_to_task(task=task_s, inner_model=net, w0=theta)
        adapt_to_task(task_s, net, theta)
        WX_q, X_q, cnnX_q, MX_q, y_q = parse_task_data(task_q)
        # # ------------------------------------------------------------------------------
        # # print output of each layer
        # w_layer_names = ['multi_graph_cnn_1', 'embedding_1', 'dense_1',\
        #                 'conv2d_1', 'multi_graph_cnn_2', 'gru_1', \
        #                 'dense_2', 'gru_2', 'cocnnattention_1', 'coattention_1',\
        #                 'batch_normalization_3', 'batch_normalization_2',\
        #                 'batch_normalization_1', 'dense_3', 'dense_4', 'dense_5',\
        #                 'dense_6', 'dense_7']
        # for name in w_layer_names:
        #     print_layer(name, net, WX_q, X_q, cnnX_q, MX_q)
        # # ------------------------------------------------------------------------------

        scores=net.evaluate([WX_q, X_q, cnnX_q, MX_q], y_q, verbose=0)

        y_pred = net.predict([WX_q, X_q, cnnX_q, MX_q])
        print("y_q:{}  y_pred:{}".format(y_q.reshape(-1)[:10], y_pred.reshape(-1)[:10]))
        # states = net.optimizer.get_weights()
        # for idx in range(len(states)):
        #     if idx==0:
        #         states[idx] = np.int64(0)
        #     else:
        #         states[idx] = np.zeros(states[idx].shape)
        # net.optimizer.set_weights(states)
        
        metrics.append(scores)
    return metrics


def get_support_query_data(task, num_training_samples_per_class):
    # # shuffle indices
    # keys = list(task.keys())
    # indices = np.arange(len(task[keys[0]]))
    # random.shuffle(indices)
    labels = list(set(task['label'].reshape(-1)))
    indices_s = []
    indices_q = []
    label_cnt = np.zeros(len(labels))
    for idx in range(len(task['label'])):
        label = task['label'][idx]
        if (label_cnt[label] < num_training_samples_per_class):
            indices_s.append(idx)
            label_cnt[label] += 1
        else:
            indices_q.append(idx)
    indices_s = np.array(indices_s)
    indices_q = np.array(indices_q)
    # print("t:{} s:{} q:{}".format(num_training_samples_per_class, len(indices_s), len(indices_q)))
    # print(indices)
    # split support and query set
    task_s = {}
    task_q = {}
    for key in task:
        task_s[key] = task[key][indices_s]
        task_q[key] = task[key][indices_q]
    return task_s, task_q

def parse_task_data(task):
    WX = np.array(task['padded_docs'])
    X = np.array(task['data_all'])
    cnnX = np.reshape(X,(-1,retweet_user_size, USER_FTS_DIM,1))
    MX = np.array(task['cos'])
    y = np.array(task['label']).astype('int')
    return WX, X, cnnX, MX, y

# print some layer's output
def print_layer(layer_name, net, WX, X, cnnX, MX):
    print(layer_name)
    layer_model = Model(inputs=net.input,outputs=net.get_layer(layer_name).output)
    layer_out = layer_model.predict([WX, X, cnnX, MX])
    print(layer_out[0])


def fetch_topic_tasks(data_idxs, accum_topic_task_num, topic_idxs):
    # print(topic_idxs)
    idxs = []
    for topic_idx in topic_idxs:
        if topic_idx == 0:
            idxs += list(data_idxs[0:accum_topic_task_num[topic_idx]])
        else:
            idxs += list(data_idxs[accum_topic_task_num[topic_idx-1]:accum_topic_task_num[topic_idx]])
    idxs = np.array(idxs)
    # print(idxs)
    return idxs



# -------------------------------------------------------------------------------------------------
# MAIN program
# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # --------------------------------------------------
    # Setup CPU or GPU
    # --------------------------------------------------
    device = torch.device("cuda")
    CNN_OUTPUT_LENGTH = int(retweet_user_size*USER_FTS_DIM/STRIDE_SIZE)
    # ----------------------------------------------
    # Setup destination folder
    # ----------------------------------------------
    dst_root_folder = 'model/abml_without_z'
    dst_folder = '{0:s}/{1:d}retweeter_seed{2:d}'.format(
        dst_root_folder,
        retweet_user_size,
        seed
    )
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # ----------------------------------------------
    # Load dataset
    # ----------------------------------------------
    # load raw dataset
    raw_dataset = load_raw_dataset(DATA_ROOT_PATH)

    # build dataset with preprocessing
    # parameter setting
    data_builder = DatasetBuilder(raw_dataset, retweet_user_size)
    dataset, topic_index = data_builder.create_dataset()
    print('Topics in dataset: {}'.format(topic_index.keys()))
    print('Dataset size: {}'.format(len(dataset)))
    # raw_task: [[0:[t_ids],1:[t_ids]],...]
    raw_tasks = [topic_index[topic] for topic in topic_index.keys()]
    task_sizes = []
    print("scaled task distribution:")
    for task in raw_tasks:
        print([len(task[key]) for key in task.keys()])
        task_sizes.append(sum([len(task[key]) for key in task.keys()]))
    task_sizes = np.array(task_sizes)

    # split tasks in the dataset for training and testing
    idxs = np.arange(0, len(raw_tasks))

    train_idxs, test_idxs = split_dataset(idxs, topic_split_rate[2], seed)

    val_idxs = train_idxs[-topic_split_rate[1]:]
    train_idxs = train_idxs[:topic_split_rate[0]]

    print("train-val-test={}-{}-{}".format(train_idxs, val_idxs, test_idxs))

    # train_idxs, val_idxs, test_idxs = split_dataset(idxs, topic_split_rate, topic_task_nums, seed)
    train_ids = [raw_tasks[train_idx] for train_idx in train_idxs]
    val_ids = [raw_tasks[val_idx] for val_idx in val_idxs]
    test_ids = [raw_tasks[test_idx] for test_idx in test_idxs]
    print("train-val-test={}-{}-{}".format(len(train_ids), len(val_ids), len(test_ids)))

    # ----------------------------------------------
    # define GCAN model
    # ----------------------------------------------
    #source tweet encoding
    # winput = Input(shape=(TEXT_LEN,))
    # wembed = Embedding(vocab_size, SOURCE_TWEET_OUTPUT_DIM, input_length=TEXT_LEN)(winput)
    # wembed = Reshape((TEXT_LEN, SOURCE_TWEET_OUTPUT_DIM))(wembed)
    # wembed = GRU(SOURCE_TWEET_OUTPUT_DIM, return_sequences=True)(wembed)
    # wembed = Dense(SOURCE_TWEET_OUTPUT_DIM, activation="tanh")(wembed)    # addition
    winput = Input(shape=(BERT_EMB_DIM,))
    wembed = Reshape((TEXT_LEN, SOURCE_TWEET_OUTPUT_DIM))(winput)
    wembed = GRU(SOURCE_TWEET_OUTPUT_DIM, return_sequences=True)(wembed)
    wembed = BatchNormalization()(wembed)

    #user propagation representation
    rmain_input = Input(shape=(retweet_user_size, USER_FTS_DIM))
    rnnencoder = GRU(OUTPUT_DIM, return_sequences=True)(rmain_input)
    rnnoutput1 = AveragePooling1D(retweet_user_size)(rnnencoder)
    rnnoutput = Flatten()(rnnoutput1)
    rnnoutput = BatchNormalization()(rnnoutput)

    #Graph-aware Propagation Representation
    graph_conv_filters_input = Input(shape=(retweet_user_size, retweet_user_size))
    gmain_input = MultiGraphCNN(GCN_OUTPUT_DIM, GCN_FILTERS_NUM)([rmain_input, graph_conv_filters_input])
    gmain_input = Dense(GCN_OUTPUT_DIM, activation="tanh")(gmain_input)  # addition
    gmain_input = MultiGraphCNN(GCN_OUTPUT_DIM, GCN_FILTERS_NUM)([gmain_input, graph_conv_filters_input])
    gmain_input = Dense(GCN_OUTPUT_DIM, activation="tanh")(gmain_input)  # addition

    # dual co attention
    gco = coattention(32, retweet_user_size)([wembed, gmain_input])
    gco = Flatten()(gco)
    gco = Dense(64, activation="tanh")(gco)    # addition1
    gco = BatchNormalization()(gco)
        
    cmain_input = Input(shape=(retweet_user_size, USER_FTS_DIM, 1))
    cnnco = Conv2D(filters=CNN_OUTPUT_DIM, kernel_size=CNN_FILTER_SIZE, strides=1, \
        activation="tanh", padding="same")(cmain_input)
    maxpooling = Reshape((CNN_OUTPUT_LENGTH, CNN_OUTPUT_DIM))(cnnco)

    co = cocnnattention(32, retweet_user_size)([wembed, maxpooling])
    co = Flatten()(co)
    co = Dense(64, activation="tanh")(co)    # addition1
    co = BatchNormalization()(co)

    merged_vector = keras.layers.concatenate([co,gco,rnnoutput])
    x = Dense(OUTPUT_DIM,activation="relu")(merged_vector)
    prediction = Dense(1,activation="sigmoid")(x)
        
    net = Model([winput,rmain_input,cmain_input, graph_conv_filters_input], prediction)
    # print the structure of GCAN
    if train_flag:
        net.summary()

    Adam=keras.optimizers.Adam(lr=inner_lr)
    net.compile(optimizer=Adam, loss="binary_crossentropy", \
        metrics=['accuracy', f1_score, precision, recall])

    if train_flag:
        net.save_weights(INNER_MODEL_PATH)


    net1 = Model([winput,rmain_input,cmain_input, graph_conv_filters_input], prediction)
    net1.compile(optimizer=keras.optimizers.Adam(lr=inner_lr), loss="binary_crossentropy", \
        metrics=['accuracy', f1_score, precision, recall])
    # ----------------------------------------------
    # Intialize/Load meta-parameters
    # ----------------------------------------------
    if resume_epoch == 0:
        loss_meta_saved = [] # to monitor the meta-loss
        loss_kl_saved = []

        # initialise meta-parameters
        all_weights = net.get_weights()
        theta = {'mean':[], 'logSigma':[]}
        for w in all_weights:
            th_m = w.copy()
            th_l = w.copy()-20
            theta['mean'].append(th_m)
            theta['logSigma'].append(th_l)
    else:
        checkpoint_filename = ('Epoch_{0:d}.pt').format(resume_epoch)
        checkpoint_file = os.path.join(dst_folder, checkpoint_filename)
        print('Start to load weights from')
        print('{0:s}'.format(checkpoint_file))
        if torch.cuda.is_available():
            saved_checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage.cuda(gpu_id))
        else:
            saved_checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        theta = saved_checkpoint['theta']

    # print(train_ids)
    # print(val_ids)
    if train_flag:
        meta_train(train_ids, val_ids, net, theta, test_ids, dataset)
        # meta_train(train_data, val_data, net, theta)
        # print("Testing...")
        # metrics = validate_classification(test_ids, net, theta, dataset, csv_flag=False)
        # # print all metrics
        # for item in metrics:
        #     print(item)
        # print("\nAverage: [loss, accuracy, f1_score, precision, recall]\n{}".format(\
        #     np.mean(metrics, axis=0)))
    else: # validation
        assert resume_epoch > 0
        metrics = validate_classification(test_ids, net, theta, dataset, csv_flag=False)
        # print all metrics
        for item in metrics:
            print(item)
        test_sizes = task_sizes[test_idxs]
        test_w = test_sizes/sum(test_sizes)
        mean_metrics = np.array(metrics).T.dot(np.array(test_w))
        print("\nAverage: [loss, accuracy, f1_score, precision, recall]\n{}".format(\
            mean_metrics))
        with open(log_file, 'a+') as f:
            f.write(dst_folder+" resume_epoch={}: {}\n".format(\
               resume_epoch, mean_metrics))
        # print(test_ids)
        # # print weights of each layer
        # w_layer_names = ['multi_graph_cnn_1', 'embedding_1', 'dense_1',\
        #                 'conv2d_1', 'multi_graph_cnn_2', 'gru_1', \
        #                 'dense_2', 'gru_2', 'cocnnattention_1', 'coattention_1',\
        #                 'batch_normalization_3', 'batch_normalization_2',\
        #                 'batch_normalization_1', 'dense_3', 'dense_4']
        # for name in w_layer_names:
        #     layer = net.get_layer(name)
        #     w = layer.get_weights()
        #     print(name)
        #     print(w[0])

        # print all the weights
        # weights = net.get_weights()
        # for w in weights:
        #     print(w[0])
