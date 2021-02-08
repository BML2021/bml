import numpy as np
from datetime import datetime, timedelta
import os
import glob
import torch
import pandas as pd
# import jieba
import jieba.analyse as analyse
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from transformers import BertTokenizer, TFBertModel
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import time
import json
import tensorflow as tf
from keras import backend as K

from cfg import DATA_ROOT_PATH

#evaluation
def f1_score(y_true, y_pred):
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

def cos_similarity(a, b):
    return a.dot(b)/(np.linalg.norm(a)*np.linalg.norm(b))

def from_date_text_to_timestamp(datestr):
    datestr = datestr.split()
    month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,\
        'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    year = int(datestr[-1])
    month = month_map[datestr[1]]
    day = int(datestr[2])
    return (datetime(year, month, day) - datetime(1970, 1, 1)) / timedelta(days=1)

def text_abstract(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    # Remove words that are only one or two character.
    docs = [[token for token in doc if len(token) > 2] for doc in docs]
    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    docs = [[token for token in doc if token not in stopwords] for doc in docs]
    # Lemmatize the documents.
    lemmatizer = WordNetLemmatizer()
    # nltk.download('wordnet')
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    full_text = ''
    for doc in docs:
        for token in doc:
            full_text += token+' '
        full_text += '\n'
    tfidf=analyse.extract_tags
    keywords=tfidf(full_text)
    return keywords[:10]

def load_raw_dataset(data_root_path):
    t1 = time.time()

    raw_dataset = {}
    threads_paths = get_threads_paths(data_root_path)
    for path in threads_paths:
        topic = path.split('/')[-1].split('-')[0]
        raw_dataset[topic] = {}
        raw_dataset[topic]['non-rumours'] = load_tweet_set(path+'/'+'non-rumours')
        raw_dataset[topic]['rumours'] = load_tweet_set(path+'/'+'rumours')
        # print(raw_dataset)
        # break # test

    t2 = time.time()
    print('load data from raw dataset: {}s'.format(t2-t1))

    return raw_dataset

def get_threads_paths(data_root_path):
    threads_paths = [data_root_path+'/'+file_name for file_name in os.listdir(data_root_path)\
        if '.' not in file_name]
    return threads_paths


def load_tweet_set(tweet_set_path):
    tweets = []
    tweets_path = [tweet_set_path+'/'+file_name for file_name in os.listdir(tweet_set_path)\
        if '.' not in file_name]
    for tweet_path in tweets_path:
        tweet = load_tweet(tweet_path)
        if tweet != None:
            tweets.append(tweet)
        # break # test
    # print(tweets_path)
    return tweets

def load_tweet(tweet_path):
    tweet_info = {}
    annotation_file = tweet_path+'/'+'annotation.json'
    structure_file = tweet_path+'/'+'structure.json'
    tweet_id = int(tweet_path.split('/')[-1])
    source_tweet_file = tweet_path+'/source-tweets/'+str(tweet_id)+'.json'
    reactions_path = tweet_path+'/reactions'
    reaction_files = [reactions_path+'/'+file_name for file_name in os.listdir(reactions_path)\
        if '_' not in file_name]
    # ensure tweets in structure == tweets in reactions + source tweet
    structure = load_json(structure_file)
    reactions = [load_json(reaction_file) for reaction_file in reaction_files]

    structure_q = tree2list(structure)
    if (len(structure_q) != len(reactions)+1):
        return None
    else:
        tweet_info['structure'] = structure
        tweet_info['reactions'] = reactions
        tweet_info['tweet_id'] = tweet_id
        tweet_info['annotation'] = load_json(annotation_file)
        tweet_info['source_tweet'] = load_json(source_tweet_file)
        return tweet_info

def load_json(json_file):
    with open(json_file, encoding='utf-8') as j_file:
        content = json.load(j_file)
    return content

# Convert the propagation tree into a queue. The first convex is the source tweet.
def tree2list(structure):
    if structure == []:
        return []
    else:
        queue = []
        for key in structure.keys():
            if (key.isdigit()):
                queue.append(int(key))
                queue += tree2list(structure[key])
        return queue

def words_embedding(topic_abstract):
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

    # raw_dataset = load_raw_dataset(DATA_ROOT_PATH)
    # id_texts = load_tweet_ids_with_text(raw_dataset)
    # ids = [id_texts[idx][0] for idx in range(len(id_texts))]
    # texts = [id_texts[idx][1] for idx in range(len(id_texts))]
    topic_abstract_text = ''
    for word in topic_abstract:
        topic_abstract_text += word+' '
    
    result = []
    input_ids = torch.tensor([tokenizer.encode(topic_abstract_text, add_special_tokens=True)])[:,:512]
    with torch.no_grad():
        outputs = model(input_ids)[1].tolist()
    return np.array(outputs)

def gen_topics_embedding(raw_dataset, raw_tasks):
    id_texts = load_tweet_ids_with_text(raw_dataset)
    id2text = {item[0]:item[1] for item in id_texts}
    topics_emb = []
    for task in raw_tasks:
        topic_texts = []
        for key in task:
            for t_id in task[key]:
                topic_texts.append(id2text[t_id])
        topic_abstract = text_abstract(topic_texts)
        topics_emb.append(words_embedding(topic_abstract))
    topics_emb = np.array(topics_emb)
    return topics_emb

def load_tweet_ids_with_text(raw_dataset):
    t1 = time.time()
    id_text_list = []
    for topic in raw_dataset.keys():
        for key in raw_dataset[topic].keys():
            for tweet in raw_dataset[topic][key]:
                text = tweet['source_tweet']['text'].split()
                clr_text = ''
                for item in text:
                    if 'http' not in item:  # delete url address
                        clr_text += item+' '
                id_text_list.append([int(tweet['tweet_id']), clr_text])
    t2 = time.time()
    print('load tweet ids and text from raw dataset: {}s'.format(t2-t1))
    return id_text_list

def get_vocab_size(raw_dataset):
    id_text_list = load_tweet_ids_with_text(raw_dataset)
    all_content = ''
    for item in id_text_list:
        all_content += ' '+item[1]
    all_content = all_content.split()
    all_content = set(all_content)
    #vocab_size: how many different words in the news content  
    vocab_size = len(all_content)
    return vocab_size

def sample_task_from_raw_task_by_label(raw_task_by_label, s_shot_per_class, q_shot_per_class, sample_flag=True):
    
    # # test
    # y = []
    # for label in raw_task_by_label:
    #     t = raw_task_by_label[label]
    #     print('r{}: {}'.format(label, len(t)))
    #     t_data = load_struct_data([t], dataset)[0]
    #     y += list(t_data['label'])
    # print('r 1:{}'.format(len([item for item in y if item == 1])))
    # print('r 0:{}'.format(len([item for item in y if item == 0])))
    # # end test
    label_list = [label for label in raw_task_by_label]
    if sample_flag:
        np.random.seed()
        sampled_task = {}
        split_points = {label:int(s_shot_per_class/(s_shot_per_class+q_shot_per_class)*len(raw_task_by_label[label])) \
            for label in label_list}
        sampled_task['s'] = []
        for label in label_list:
            if len(raw_task_by_label[label]) != 0:
                sampled_task['s'] += list(np.random.choice(raw_task_by_label[label][:split_points[label]], size=s_shot_per_class, \
                    replace=True, p=None))
        sampled_task['s'] = np.array(sampled_task['s'])
        np.random.shuffle(sampled_task['s'])

        sampled_task['q'] = []
        for label in label_list:
            if len(raw_task_by_label[label]) != 0:
                sampled_task['q'] += list(np.random.choice(raw_task_by_label[label][split_points[label]:], size=q_shot_per_class, \
                    replace=True, p=None))
        sampled_task['q'] = np.array(sampled_task['q'])
        np.random.shuffle(sampled_task['q'])
    else:
        # no sample
        sampled_task = {}
        sampled_task['s'] = []
        for label in label_list:
            if len(raw_task_by_label[label]) != 0:
                sampled_task['s'] += list(raw_task_by_label[label][:s_shot_per_class])
        sampled_task['s'] = np.array(sampled_task['s'])
        np.random.shuffle(sampled_task['s'])

        sampled_task['q'] = []
        for label in label_list:
            if len(raw_task_by_label[label]) != 0:
                sampled_task['q'] += list(raw_task_by_label[label][s_shot_per_class:])
        sampled_task['q'] = np.array(sampled_task['q'])
        np.random.shuffle(sampled_task['q'])
    return sampled_task
        

def load_struct_data(tasks_ids, dataset):
    # label_stats = {0:0,1:0}
    data = []
    for task_ids in tasks_ids:
        task_data = {'data_all':[], 'padded_docs':[], 'cos':[], 'label':[]}
        for news_id in task_ids:
            task_data['data_all'].append(dataset[news_id]['data_all'])
            task_data['padded_docs'].append(dataset[news_id]['padded_docs'])
            task_data['cos'].append(dataset[news_id]['cos'])
            task_data['label'].append(dataset[news_id]['label'])
            # label_stats[dataset[news_id]['label']] += 1
        for key in task_data:
            task_data[key] = np.array(task_data[key])
        data.append(task_data)
    # print(label_stats)
    return data

def split_dataset(idxs, test_size, seed):
    dataset_size = len(idxs)
    np.random.seed(seed)
    np.random.shuffle(idxs)
    return idxs[:-test_size], idxs[-test_size:]

# print some layer's output
def print_layer(layer_name, net, WX, X, cnnX, MX):
    print(layer_name)
    layer_model = Model(inputs=net.input,outputs=net.get_layer(layer_name).output)
    layer_out = layer_model.predict([WX, X, cnnX, MX])
    print(layer_out[0])

 


