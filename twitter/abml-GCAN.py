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
from dataset import DatasetBuilder, get_vocab_size, TEXT_LEN, USER_FTS_DIM
from task_generator import load_raw_data, gen_topics, gen_raw_tasks, sample_task_from_raw_task, \
    gen_raw_tasks_by_label, sample_task_from_raw_task_by_label, load_struct_data, text_preprocess,\
    train_word2vec, gen_topics_embedding, \
    DATA_DIR, WORD2VEC_MODEL_PATH, WORD_DIM, NO_BELOW, NO_ABOVE, MIN_COUNT
from utils import cos_similarity

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
parser.add_argument('--topic_split_rate', type=list, default=[0.7,0.1,0.2], \
    help='Dataset split ratio')
parser.add_argument('--seed', type=int, default=10, help='Seed for splitting train-test set')
parser.add_argument('--num_val_shots', type=int, default=3, help='Number of validation samples per class')
parser.add_argument('--sample_rate', type=int, default=4, help='Rate for sampling tasks from topics')
parser.add_argument('--batch_size', type=int, default=1, help='Number of tasks per batch')
parser.add_argument('--batch_per_epoch', type=int, default=6, help='Number of batches per epoch')
parser.add_argument('--retweet_user_size', type=int, default=30, help='The length of retweet propagation you want to utilize')
parser.add_argument('--data_scale', type=float, default=1.0, help='The ratio of data used in experiment')


parser.add_argument('--inner_lr', type=float, default=2e-3, help='Learning rate for task adaptation')
parser.add_argument('--num_inner_updates', type=int, default=3, help='Number of gradient updates for task adaptation')
parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta-parameters')
parser.add_argument('--lr_decay', type=float, default=0.98, help='Decay factor of meta-learning rate (<=1), 1 = no decay')
# parser.add_argument('--minibatch', type=int, default=5, help='Number of tasks per minibatch to update meta-parameters')

parser.add_argument('--num_epochs', type=int, default=20, help='How many 10,000 tasks are used to train?')
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')


parser.add_argument('--log_file', type=str, default="log.txt", help='Name of log file')

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

data_scale = args.data_scale
print("The ratio of data used in experiment:{}".format(data_scale))


num_epochs_save = 1

num_epochs = args.num_epochs

resume_epoch = args.resume_epoch

sample_rate = args.sample_rate

lr_decay = args.lr_decay
print('Learning rate decay = {0}'.format(lr_decay))

log_file = args.log_file
print("Log file: {}".format(log_file))

#-------------------------
# parameters setting
#-------------------------
SOURCE_TWEET_OUTPUT_DIM = 32
OUTPUT_DIM = 64 
GCN_FILTERS_NUM = 1
GCN_OUTPUT_DIM = 32
CNN_FILTER_SIZE = (3,3)
CNN_OUTPUT_DIM = 32
STRIDE_SIZE = 1
CNN_OUTPUT_LENGTH = int(retweet_user_size*USER_FTS_DIM/STRIDE_SIZE)
MIN_EPOCH_NUM = 50
MAX_EPOCH_NUM = 100

if not os.path.exists("model/inner_model"):
    os.makedirs("model/inner_model")
INNER_MODEL_PATH = "model/inner_model/abml_z_{}retweeter.h5".format(retweet_user_size)
# --------------------------------------------------
# Setup CPU or GPU
# --------------------------------------------------
device = torch.device("cpu")



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


def meta_train(train_idxs, val_idxs, test_idxs, raw_tasks, topics_emb, net, theta, dataset):
    train_ids = [raw_tasks[train_idx] for train_idx in train_idxs]
    val_ids = [raw_tasks[val_idx] for val_idx in val_idxs]
    test_ids = [raw_tasks[test_idx] for test_idx in test_idxs]

    train_embs = [topics_emb[train_idx] for train_idx in train_idxs]
    val_embs = [topics_emb[val_idx] for val_idx in val_idxs]
    test_embs = [topics_emb[test_idx] for test_idx in test_idxs]

    train_emb_distrib = compute_mix_emb(train_ids, train_embs)

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
            batch_embs = [train_embs[batch_idx] for batch_idx in batch_idxs]
            for idx in range(len(batch_ids)):
                task_ids = batch_ids[idx]
                # run a task
                # load data of a task
                net.load_weights(INNER_MODEL_PATH)
                # task_s, task_q = get_support_query_data(train_task, num_training_samples_per_class)
                # print(task_ids)
                task_sample_ids = sample_task_from_raw_task_by_label(task_ids, support_shots, query_shots)

                task_s = load_struct_data([task_sample_ids['s']], dataset)[0]
                task_q = load_struct_data([task_sample_ids['q']], dataset)[0]

                zeta = cos_similarity(train_emb_distrib, batch_embs[idx])
                drop_theta(zeta, theta)

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
        metrics = validate_classification(train_idxs, val_idxs, raw_tasks, topics_emb, net, theta, dataset)
        for score in metrics:
            print(score)
        print("\nVal average: [loss, accuracy, f1_score, precision, recall]\n{}\n".format(\
                np.mean(metrics, axis=0)))
        with open(log_file, 'a+') as f:
            f.write("Val average epoch={}: {}\n".format(\
               epoch+1, np.mean(metrics, axis=0)))


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
    w0 = sample_nn_weight(theta)
    inner_model.set_weights(w0)
    inner_model.fit([WX, X, cnnX, MX], y, epochs=num_inner_updates, verbose=0)
    # y_pred = inner_model.predict([WX, X, cnnX, MX])

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

# validate_classification(test_idxs, raw_tasks, topics_emb, net, theta, dataset, csv_flag=False)
def validate_classification(train_idxs, test_idxs, raw_tasks, topics_emb, net, theta, dataset):
    train_ids = [raw_tasks[train_idx] for train_idx in train_idxs]
    test_ids = [raw_tasks[test_idx] for test_idx in test_idxs]

    train_embs = [topics_emb[train_idx] for train_idx in train_idxs]
    test_embs = [topics_emb[test_idx] for test_idx in test_idxs]

    train_emb_distrib = compute_mix_emb(train_ids, train_embs)

    support_shots = num_training_samples_per_class
    query_shots = num_classes_per_task
    # support_shots = num_training_samples_per_class * sample_rate
    # query_shots = num_classes_per_task * sample_rate

    metrics = []
    
    # total_val_samples_per_task = num_val_samples_per_class * num_classes_per_task
    for idx in range(len(test_ids)):
        task_ids = test_ids[idx]
        task_embs = test_embs[idx]

        net.load_weights(INNER_MODEL_PATH)
        # task_s, task_q = get_support_query_data(task, num_training_samples_per_class)
        # task_sample_ids = sample_task_from_raw_task(task_ids, support_shots, query_shots)
        task_sample_ids = sample_task_from_raw_task_by_label(task_ids, support_shots, query_shots, False)
        task_s = load_struct_data([task_sample_ids['s']], dataset)[0]
        task_q = load_struct_data([task_sample_ids['q']], dataset)[0]

        zeta = cos_similarity(train_emb_distrib, task_embs)

        drop_theta(zeta, theta)

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
        print("y_q:{}  y_pred:{}".format(y_q.reshape(-1), y_pred.reshape(-1)))
        
        metrics.append(scores)

        # if not train_flag:
        #     sys.stdout.write('\033[F')
    return metrics


def get_support_query_data(task, num_training_samples_per_class):

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
    cnnX = np.reshape(X,(-1, retweet_user_size, USER_FTS_DIM,1))
    MX = np.array(task['cos'])
    y = np.array(task['label']).astype('int')
    return WX, X, cnnX, MX, y

# print some layer's output
def print_layer(layer_name, net, WX, X, cnnX, MX):
    print(layer_name)
    layer_model = Model(inputs=net.input,outputs=net.get_layer(layer_name).output)
    layer_out = layer_model.predict([WX, X, cnnX, MX])
    print(layer_out[0])

def split_dataset(idxs, test_size, fold_state):
    dataset_size = len(idxs)
    assert fold_state < int(1/test_size)

    idxs_begin = int(test_size*dataset_size*fold_state)
    idxs_end = int(test_size*dataset_size*(fold_state+1))
    # print(test_size, dataset_size, fold_state)
    # print("begin-end:{}-{}".format(idxs_begin, idxs_end))

    test_idxs = idxs[idxs_begin:idxs_end]
    if idxs_begin == 0:
        train_idxs = idxs[idxs_end:]
    elif idxs_end == dataset_size:
        train_idxs = idxs[:idxs_begin]
    else:
        train_idxs = np.concatenate((idxs[:idxs_begin], idxs[idxs_end:]),axis=0)
    return train_idxs, test_idxs

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

def compute_mix_emb(train_ids, train_embs):
    weights = []
    for topic_ids in train_ids:
        weight = 0
        for label in topic_ids:
            weight += len(topic_ids[label])
        weights.append(weight)
    weights = np.array(weights)
    weights = weights/sum(weights)
    train_emb_distrib = np.zeros(train_embs[0].shape)
    for idx in range(len(train_ids)):
        train_emb_distrib += weights[idx]*train_embs[idx]

    return train_emb_distrib

def drop_theta(zeta, theta):
    theta_rand = sample_theta(theta)
    theta['mean'] = [theta['mean'][i]*zeta+(1-zeta)*theta_rand['mean'][i] for i in range(len(theta['mean']))]
    theta['logSigma'] = [theta['logSigma'][i]*zeta+(1-zeta)*theta_rand['logSigma'][i] for i in range(len(theta['logSigma']))]

def sample_theta(theta):
    np.random.seed()
    theta_rand = {'mean':[], 'logSigma':[]}
    for i in range(len(theta['mean'])):
        th_m = np.random.normal(0.0, 0.2, theta['mean'][i].shape)
        theta_rand['mean'].append(theta['mean'][i]*(1+th_m))
        theta_rand['logSigma'].append(theta['logSigma'][i]*(1+th_m))
    return theta_rand

# -------------------------------------------------------------------------------------------------
# MAIN program
# -------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # ----------------------------------------------
    # Setup destination folder
    # ----------------------------------------------
    dst_root_folder = 'model/abml_z'
    dst_folder = '{0:s}/{1:s}_{2:d}retweeter_{3}ratio_seed{4:d}'.format(
        dst_root_folder,
        datasource,
        retweet_user_size,
        data_scale,
        seed
    )
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # --------------------------------------------------
    #   Setup based model/network
    # --------------------------------------------------
    if datasource in ['twitter15', 'twitter16']:

        # ----------------------------------------------
        # Load dataset
        # ----------------------------------------------
        dataset_dir = os.path.join(DATA_DIR, datasource)
        ids, docs, labels = load_raw_data(dataset_dir)
        docs_text, corpus, dictionary = text_preprocess(docs, NO_BELOW, NO_ABOVE, MIN_COUNT)
        topics = gen_topics(corpus, num_topics, dictionary, load_model=True)
        w2v_model = train_word2vec(docs_text, WORD2VEC_MODEL_PATH, WORD_DIM, load_model=True)
        topics_emb = gen_topics_embedding(corpus, num_topics, dictionary, WORD_DIM, w2v_model)
        raw_tasks, _ = gen_raw_tasks_by_label(np.array(ids), topics, labels, num_classes_per_task, \
            num_samples_per_class)

        print("Tasks len:", len(raw_tasks))
        if not os.path.exists(DATA_DIR+"/"+datasource+"/id_index"):
            os.makedirs(DATA_DIR+"/"+datasource+"/id_index")
        if os.path.exists(DATA_DIR+"/"+datasource+"/id_index/processed_dataset_{}.txt".format(retweet_user_size)):
            with open(DATA_DIR+"/"+datasource+"/id_index/processed_dataset_{}.txt".format(retweet_user_size), "r") as f:
                dataset = f.read()
                dataset = eval(dataset)
            vocab_size = get_vocab_size(datasource)
        else:
            data_builder = DatasetBuilder(datasource, retweet_user_size, time_cutoff=None, only_binary=True)
            dataset = data_builder.create_dataset(dataset_type="id_index", standardize_features=True)
            vocab_size = data_builder.get_vocab_size()
            np.set_printoptions(threshold=1e6)
            with open(DATA_DIR+"/"+datasource+"/id_index/processed_dataset_{}.txt".format(retweet_user_size), "w") as f:
                f.write(str(dataset))
        print("dataset size: {}".format(len(dataset)))

        # split dataset for training and testing
        idxs = np.arange(0, len(raw_tasks))

        train_idxs, test_idxs = split_dataset(idxs, topic_split_rate[2], seed)
        
        test_num = 1
        val_num = 1
        train_num = int(len(raw_tasks)*data_scale)-test_num-val_num

        val_idxs = train_idxs[-val_num:]
        train_idxs = train_idxs[:train_num]
        test_idxs = test_idxs[:test_num]

        print("train-val-test={}-{}-{}".format(len(train_idxs), len(val_idxs), len(test_idxs)))

        # ----------------------------------------------
        # define GCAN model
        # ----------------------------------------------
        #source tweet encoding
        winput = Input(shape=(TEXT_LEN,))
        wembed = Embedding(vocab_size, SOURCE_TWEET_OUTPUT_DIM, input_length=TEXT_LEN)(winput)
        wembed = Reshape((TEXT_LEN, SOURCE_TWEET_OUTPUT_DIM))(wembed)
        wembed = GRU(SOURCE_TWEET_OUTPUT_DIM, return_sequences=True)(wembed)
        wembed = Dense(SOURCE_TWEET_OUTPUT_DIM, activation="tanh")(wembed)    # addition

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


        if train_flag:
            meta_train(train_idxs, val_idxs, test_idxs, raw_tasks, topics_emb, net, theta, dataset)

        else: # validation
            assert resume_epoch > 0

            metrics = validate_classification(train_idxs, test_idxs, raw_tasks, topics_emb, net, theta, dataset)

            for item in metrics:
                print(item)
            print("\nAverage: [loss, accuracy, f1_score, precision, recall]\n{}".format(\
                np.mean(metrics, axis=0)))
            with open(log_file, 'a+') as f:
                f.write(dst_folder+" resume_epoch={}: {}\n".format(\
                   resume_epoch, np.mean(metrics, axis=0)))



    else:
        sys.exit('Unknown dataset!')
