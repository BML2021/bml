# coding: utf-8
# In[ ]:

import numpy as np
from numpy import *
import tensorflow as tf
from sklearn.model_selection import train_test_split
import torch
import os
import keras
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Embedding, Conv2D, Dense, Reshape, GRU, \
                         AveragePooling1D, Flatten, BatchNormalization
from keras.models import Model
import argparse

from GCAN import preprocess_adj_tensor, MultiGraphCNN, coattention, cocnnattention
from dataset import DatasetBuilder, get_vocab_size, TEXT_LEN, USER_FTS_DIM
from task_generator import load_raw_data, gen_topics, gen_raw_tasks, sample_task_from_raw_task, \
    load_struct_data, text_preprocess, \
    DATA_DIR, NO_BELOW, NO_ABOVE, MIN_COUNT


#using cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------------------------------------
# Setup input parser
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables for MAML.')
parser.add_argument('--datasource', type=str, default='twitter15', help='Datasource: twitter15, twitter16')
parser.add_argument('--fold_state', type=int, default=0, help='State to split dataset')
parser.add_argument('--log_file', type=str, default="log.txt", help='Name of log file')
parser.add_argument('--retweet_user_size', type=int, default=30, help='The length of retweet propagation you want to utilize')
parser.add_argument('--data_scale', type=float, default=1.0, help='The ratio of data used in experiment')

args = parser.parse_args()
datasource = args.datasource
print('Dataset = {0:s}'.format(datasource))
fold_state = args.fold_state
print('Fold state = {}'.format(fold_state))
log_file = args.log_file
print("Log file: {}".format(log_file))
retweet_user_size = args.retweet_user_size
print("The length of retweet propagation:{}".format(retweet_user_size))
data_scale = args.data_scale
print("The ratio of data used in experiment:{}".format(data_scale))

#parameters setting
SOURCE_TWEET_OUTPUT_DIM = 32
OUTPUT_DIM = 64 
GCN_FILTERS_NUM = 1
GCN_OUTPUT_DIM = 32
CNN_FILTER_SIZE = (3,3)
CNN_OUTPUT_DIM = 32
STRIDE_SIZE = 1
CNN_OUTPUT_LENGTH = int(retweet_user_size*USER_FTS_DIM/STRIDE_SIZE)
MIN_EPOCH_NUM = 20
MAX_EPOCH_NUM = 100

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

def get_w_shape(w):
    if isinstance(w, (np.ndarray)):
        print(w.shape)
    else:
        print(len(w))
        print("[")
        for item in w:
            get_w_shape(item)
        print("]")

def load_ids_from_tasks(tasks):
    ids = []
    for task in tasks:
        ids += task
    ids = np.array(ids)
    np.random.seed()
    np.random.shuffle(ids)
    return ids

def parse_task_data(task):
    WX = np.array(task['padded_docs'])
    X = np.array(task['data_all'])
    cnnX = np.reshape(X,(-1,retweet_user_size, USER_FTS_DIM,1))
    MX = np.array(task['cos'])
    y = np.array(task['label']).astype('int')
    return WX, X, cnnX, MX, y

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

if __name__ == '__main__':


    num_topics = 10
    num_classes_per_task = 2
    num_training_samples_per_class = 3
    seed = 10
    topic_split_rate = [0.8, 0.2]
    # ----------------------------------------------
    # Load dataset
    # ----------------------------------------------
    dataset_dir = os.path.join(DATA_DIR, datasource)
    ids, docs, labels = load_raw_data(dataset_dir)
    docs_text, corpus, dictionary = text_preprocess(docs, NO_BELOW, NO_ABOVE, MIN_COUNT)
    topics = gen_topics(corpus, num_topics, dictionary, load_model=True)
    # tasks_ids: list of tasks
    # task_ids: list of tweet ids in the task, which contains 
    #           num_classes_per_task*num_samples_per_class samples
    # tasks_ids, topic_task_nums, _ = gen_tasks(np.array(ids), topics, labels, \
    #     num_classes_per_task, num_samples_per_class, seed, sample_rate)
    raw_tasks, _ = gen_raw_tasks(np.array(ids), topics, labels, num_classes_per_task, \
        num_training_samples_per_class+1, seed)
    # tasks = []
    # for raw_task in raw_tasks:
    #     task = sample_task_from_raw_task(raw_task, support_shots, query_shots)
    #     tasks.append(task)
    print("Tasks len:", len(raw_tasks))
    for raw_task in raw_tasks:
        print(len(raw_task))

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
    # print("task ids shape:\n{}".format(tasks_ids.shape))
    # split dataset for training and testing
    idxs = np.arange(0,len(raw_tasks))
    # train_idxs, test_idxs = train_test_split(idxs, test_size=topic_split_rate[1], random_state=seed)
    train_idxs, test_idxs = split_dataset(idxs, topic_split_rate[1], fold_state)
        
    test_num = max(int(len(raw_tasks)*data_scale*topic_split_rate[1]), 1)
    train_num = int(len(raw_tasks)*data_scale)-test_num

    train_idxs = train_idxs[:train_num]
    test_idxs = test_idxs[:test_num]
    print(train_idxs, test_idxs)

    train_tasks = raw_tasks[train_idxs]
    train_ids = load_ids_from_tasks(train_tasks)
    test_tasks = raw_tasks[test_idxs]
    test_ids = load_ids_from_tasks(test_tasks)
    
    np.set_printoptions(threshold=1e6)
    dataset_ids = [train_ids, test_ids]
    with open('{}.txt'.format(fold_state), 'w') as f:
        f.write(str(dataset_ids))

    print("train-test={}-{}".format(len(train_ids), len(test_ids)))

    train_data = load_struct_data([train_ids], dataset)[0]
    test_data = load_struct_data([test_ids], dataset)[0]

    WX_train, X_train, cnnX_train, MX_train, y_train = parse_task_data(train_data)
    WX_test, X_test, cnnX_test, MX_test, y_test = parse_task_data(test_data)

    ##training GCAN model
    #source tweet encoding
    winput=Input(shape=(TEXT_LEN,))
    wembed=Embedding(vocab_size, SOURCE_TWEET_OUTPUT_DIM, input_length=TEXT_LEN)(winput)
    wembed=Reshape((TEXT_LEN, SOURCE_TWEET_OUTPUT_DIM))(wembed)
    wembed=GRU(SOURCE_TWEET_OUTPUT_DIM, return_sequences=True)(wembed)
    # wembed = Dense(SOURCE_TWEET_OUTPUT_DIM, activation="sigmoid")(wembed)    # addition

    #user propagation representation
    rmain_input =Input(shape=(retweet_user_size, USER_FTS_DIM))
    rnnencoder=GRU(OUTPUT_DIM, return_sequences=True)(rmain_input)
    rnnoutput1= AveragePooling1D(retweet_user_size)(rnnencoder)
    rnnoutput=Flatten()(rnnoutput1)
    rnnoutput = BatchNormalization()(rnnoutput)

    #Graph-aware Propagation Representation
    graph_conv_filters_input = Input(shape=(retweet_user_size, retweet_user_size))
    gmain_input= MultiGraphCNN(GCN_OUTPUT_DIM, GCN_FILTERS_NUM)([rmain_input, graph_conv_filters_input])
    # gmain_input = Dense(GCN_OUTPUT_DIM, activation="sigmoid")(gmain_input)  # addition
    gmain_input= MultiGraphCNN(GCN_OUTPUT_DIM, GCN_FILTERS_NUM)([gmain_input, graph_conv_filters_input])
    # gmain_input = Dense(GCN_OUTPUT_DIM, activation="sigmoid")(gmain_input)  # addition

    #dual co attention
    gco=coattention(32, retweet_user_size)([wembed, gmain_input])
    gco=Flatten()(gco)
    # gco = Dense(64, activation="sigmoid")(gco)    # addition1
    gco = BatchNormalization()(gco)
        
    cmain_input=Input(shape=(retweet_user_size, USER_FTS_DIM, 1))
    cnnco=Conv2D(filters=CNN_OUTPUT_DIM, kernel_size=CNN_FILTER_SIZE, strides=1, \
        activation="sigmoid", padding="same")(cmain_input)

    maxpooling=Reshape((CNN_OUTPUT_LENGTH, CNN_OUTPUT_DIM))(cnnco)
    
    # print(wembed.shape, maxpooling.shape)
    co=cocnnattention(32, retweet_user_size)([wembed, maxpooling])
    co=Flatten()(co)
    # co = Dense(64, activation="sigmoid")(co)    # addition1
    co = BatchNormalization()(co)

    merged_vector=keras.layers.concatenate([co,gco,rnnoutput])
    x=Dense(OUTPUT_DIM,activation="relu")(merged_vector)
    # x=Dense(OUTPUT_DIM,activation="sigmoid")(x)
    # x=Dense(OUTPUT_DIM,activation="sigmoid")(x)
    prediction=Dense(1,activation="sigmoid")(x)
        
    model=Model([winput,rmain_input,cmain_input, graph_conv_filters_input], prediction)
    model.summary()
    # weights = np.array(model.get_weights())
    # print("weights shape:{}".format(weights.shape))
    print("weights shape:")
    all_weights = model.get_weights()
    # print(get_w_shape(all_weights))

    Adam=keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer=Adam, loss="binary_crossentropy", \
        metrics=['accuracy', f1_score, precision, recall])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    history=model.fit([WX_train, X_train, cnnX_train, MX_train],\
        y_train, epochs=MIN_EPOCH_NUM, validation_split=0.125)
    history=model.fit([WX_train, X_train, cnnX_train, MX_train],\
        y_train, epochs=MAX_EPOCH_NUM-MIN_EPOCH_NUM, validation_split=0.125, callbacks=[early_stopping])
    # history=model.fit([WX_train, X_train, cnnX_train, MX_train],\
    #     y_train, epochs=MAX_EPOCH_NUM, validation_split=0.125, callbacks=[early_stopping])
    
    scores=model.evaluate([WX_test, X_test, cnnX_test, MX_test], y_test, verbose=0)
    print(scores)
    with open(log_file, 'a+') as f:
        f.write(str(scores))
        f.write('\n')
    
