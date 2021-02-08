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
from dataset import DatasetBuilder
# from task_generator import load_raw_data, gen_topics, gen_raw_tasks, sample_task_from_raw_task, \
#     gen_raw_tasks_by_label, load_struct_data, text_preprocess, train_word2vec, gen_topics_embedding,\
#     DATA_DIR, NO_BELOW, NO_ABOVE, MIN_COUNT, WORD2VEC_MODEL_PATH, WORD_DIM

from utils import f1_score, precision, recall, load_raw_dataset, load_struct_data
from cfg import *

#using cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --------------------------------------------------
# Setup input parser
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables for MAML.')
# parser.add_argument('--datasource', type=str, default='twitter15', help='Datasource: twitter15, twitter16')
parser.add_argument('--fold_state', type=int, default=0, help='State to split dataset')
parser.add_argument('--log_file', type=str, default="log_gcan.txt", help='Name of log file')
parser.add_argument('--retweet_user_size', type=int, default=25, help='The length of retweet propagation you want to utilize')
parser.add_argument('--data_scale', type=float, default=1.0, help='The ratio of data used in experiment')

args = parser.parse_args()
# datasource = args.datasource
# print('Dataset = {0:s}'.format(datasource))
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
MIN_EPOCH_NUM = 10
MAX_EPOCH_NUM = 100

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
        for key in task.keys():
            ids += task[key]
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

# def split_dataset(idxs, test_size, fold_state):
#     dataset_size = len(idxs)
#     assert fold_state < int(1/test_size)

#     idxs_begin = int(test_size*dataset_size*fold_state)
#     idxs_end = int(test_size*dataset_size*(fold_state+1))
#     # print(test_size, dataset_size, fold_state)
#     # print("begin-end:{}-{}".format(idxs_begin, idxs_end))

#     test_idxs = idxs[idxs_begin:idxs_end]
#     if idxs_begin == 0:
#         train_idxs = idxs[idxs_end:]
#     elif idxs_end == dataset_size:
#         train_idxs = idxs[:idxs_begin]
#     else:
#         train_idxs = np.concatenate((idxs[:idxs_begin], idxs[idxs_end:]),axis=0)
#     return train_idxs, test_idxs
def split_dataset(idxs, test_size, fold_state):
    dataset_size = len(idxs)
    np.random.seed(fold_state)
    np.random.shuffle(idxs)
    return idxs[:-test_size], idxs[-test_size:]

def data_scaling(raw_tasks, data_scale):
    if data_scale == 1.0:
        return np.array(raw_tasks)
    else:
        new_raw_tasks = []
        for raw_task in raw_tasks:
            keys = raw_task.keys()
            new_raw_task = {key:raw_task[key][:int(len(raw_task[key])*data_scale)] for key in keys}
            new_raw_tasks.append(new_raw_task)
        new_raw_tasks = np.array(new_raw_tasks)
        return new_raw_tasks

if __name__ == '__main__':

    


    # num_topics = 5
    # num_classes_per_task = 2
    # num_training_samples_per_class = 3
    # num_val_samples_per_class = 3
    # num_samples_per_class = num_training_samples_per_class + num_val_samples_per_class
    # seed = 5
    topic_split_rate = [7, 2]
    # # ----------------------------------------------
    # # Load dataset
    # # ----------------------------------------------
    # dataset_dir = os.path.join(DATA_DIR, datasource)
    # ids, docs, labels = load_raw_data(dataset_dir)
    # # add_docs = load_text_file(add_text_file)
    # add_docs = []
    # docs_text, corpus, dictionary = text_preprocess(docs+add_docs, NO_BELOW, NO_ABOVE, MIN_COUNT)
    # topics = gen_topics(corpus, num_topics, dictionary, load_model=True)
    # w2v_model = train_word2vec(docs_text, WORD2VEC_MODEL_PATH, WORD_DIM, load_model=True)
    # topics_emb = gen_topics_embedding(corpus, num_topics, dictionary, WORD_DIM, w2v_model)
    # raw_tasks, _ = gen_raw_tasks_by_label(np.array(ids), topics, labels, num_classes_per_task, \
    #     num_samples_per_class)
    # # tasks = []
    # # for raw_task in raw_tasks:
    # #     task = sample_task_from_raw_task(raw_task, support_shots, query_shots)
    # #     tasks.append(task)
    # print("Tasks num:", len(raw_tasks))



    # if os.path.exists(DATA_DIR+"/"+datasource+"/id_index/processed_dataset_{}.txt".format(retweet_user_size)):
    #     with open(DATA_DIR+"/"+datasource+"/id_index/processed_dataset_{}.txt".format(retweet_user_size), "r") as f:
    #         dataset = f.read()
    #         dataset = eval(dataset)
    #     vocab_size = get_vocab_size(datasource)
    # else:
    #     data_builder = DatasetBuilder(datasource, retweet_user_size, time_cutoff=None, only_binary=True)
    #     dataset = data_builder.create_dataset(dataset_type="id_index", standardize_features=True)
    #     vocab_size = data_builder.get_vocab_size()
    #     np.set_printoptions(threshold=1e6)
    #     with open(DATA_DIR+"/"+datasource+"/id_index/processed_dataset_{}.txt".format(retweet_user_size), "w") as f:
    #         f.write(str(dataset))
    # print("dataset size: {}".format(len(dataset)))
    # # print("task ids shape:\n{}".format(tasks_ids.shape))
    # # split dataset for training and testing
    # idxs = np.arange(0,len(raw_tasks))
    # # train_idxs, test_idxs = train_test_split(idxs, test_size=topic_split_rate[1], random_state=seed)
    # train_idxs, test_idxs = split_dataset(idxs, topic_split_rate[1], fold_state)
        
    # print(train_idxs, test_idxs)

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
    

    # split tasks in the dataset for training and testing
    idxs = np.arange(0, len(raw_tasks))

    train_idxs, test_idxs = split_dataset(idxs, topic_split_rate[1], fold_state)

    # val_idxs = train_idxs[-topic_split_rate[1]:]
    # train_idxs = train_idxs[:topic_split_rate[0]]

    # print("train-val-test={}-{}-{}".format(train_idxs, val_idxs, test_idxs))
    print("train-test={}-{}".format(train_idxs, test_idxs))

    #scale data in tasks
    raw_tasks = data_scaling(raw_tasks, data_scale)
    print("scaled task distribution:")
    for task in raw_tasks:
        print([len(task[key]) for key in task.keys()])


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
    # winput=Input(shape=(TEXT_LEN,))
    # wembed=Embedding(vocab_size, SOURCE_TWEET_OUTPUT_DIM, input_length=TEXT_LEN)(winput)
    # wembed=Reshape((TEXT_LEN, SOURCE_TWEET_OUTPUT_DIM))(wembed)
    # wembed=GRU(SOURCE_TWEET_OUTPUT_DIM, return_sequences=True)(wembed)
    # # wembed = Dense(SOURCE_TWEET_OUTPUT_DIM, activation="sigmoid")(wembed)    # addition
    winput = Input(shape=(BERT_EMB_DIM,))
    wembed = Reshape((TEXT_LEN, SOURCE_TWEET_OUTPUT_DIM))(winput)
    wembed = GRU(SOURCE_TWEET_OUTPUT_DIM, return_sequences=True)(wembed)
    wembed = BatchNormalization()(wembed)

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

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=2)
    history=model.fit([WX_train, X_train, cnnX_train, MX_train],\
        y_train, epochs=MIN_EPOCH_NUM, validation_split=0.2)
    history=model.fit([WX_train, X_train, cnnX_train, MX_train],\
        y_train, epochs=MAX_EPOCH_NUM-MIN_EPOCH_NUM, validation_split=0.2, callbacks=[early_stopping])
    # history=model.fit([WX_train, X_train, cnnX_train, MX_train],\
    #     y_train, epochs=MAX_EPOCH_NUM, validation_split=0.125, callbacks=[early_stopping])
    
    scores=model.evaluate([WX_test, X_test, cnnX_test, MX_test], y_test, verbose=0)
    print(scores)
    with open(log_file, 'a+') as f:
        f.write(str(scores))
        f.write('\n')
    