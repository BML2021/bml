"""
https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda
.html#sphx-glr-auto-examples-tutorials-run-lda-py
"""
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from pprint import pprint
import numpy as np
import os
from sklearn.model_selection import train_test_split
import argparse

from utils import get_tree_file_names, get_root_id, cos_similarity
from dataset import DatasetBuilder

DATA_DIR = "../maml-GCAN/rumor_detection_acl2017"
MODEL_PATH = "model/lda.model"
WORD2VEC_MODEL_PATH = "model/word2vec.model"
if not os.path.exists("model"):
        os.makedirs("model")

NO_BELOW = 5
NO_ABOVE = 0.1
MIN_COUNT = 20
WORD_DIM = 100




def gen_topics(corpus, num_topics, dictionary, load_model=True):
    # print(corpus[:5])
    # Train LDA model
    model = train_lda(load_model, corpus, num_topics, dictionary)
    
    print("Topics descriptions:")
    tps_desc = model.print_topics()  #return all the topics
    for item in tps_desc:
        print(item)


    topics = np.zeros(len(corpus)).astype('int')
    for idx in range(len(corpus)):
        topics[idx] = get_doc_topic(model, corpus[idx])
    # print(topics)
    return topics

def text_preprocess(docs, no_below, no_above, min_count):
    # text preprocess
    #word filter's parameters

    # Tokenize the documents.
    # Split the documents into tokens.
    tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()  # Convert to lowercase.
        docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]
    # Remove stopwords
    stopwords = set(nltk.corpus.stopwords.words('english'))
    docs = [[token for token in doc if token not in stopwords] for doc in docs]

    # Lemmatize the documents.
    lemmatizer = WordNetLemmatizer()
    # nltk.download('wordnet')
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]

    # Compute bigrams.
    # Add bigrams and trigrams to docs (only ones that appear min_count times or more).
    bigram = Phrases(docs, min_count=min_count)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)

    # Remove rare and common tokens.
    # Create a dictionary representation of the documents.
    dictionary = Dictionary(docs)

    # Filter out words that occur less than no_below documents, or more than no_above of the documents.
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    return docs, corpus, dictionary

def train_lda(load_model, corpus, num_topics, dictionary):
    # Train LDA model.
    if load_model and os.path.exists(MODEL_PATH):
        model = LdaModel.load(MODEL_PATH)
    else:
        # Set training parameters.
        chunksize = 2000
        passes = 20 #epoch number
        iterations = 400
        eval_every = None  # Don't evaluate model perplexity, takes too much time.
        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        model = LdaModel(
            corpus=corpus,
            id2word=id2word,
            chunksize=chunksize,
            alpha='auto',
            eta='auto',
            iterations=iterations,
            num_topics=num_topics,
            passes=passes,
            eval_every=eval_every
        )
        model.save(MODEL_PATH)
    return model

def get_doc_topic(model, doc_bow):
    topics_prob = model.get_document_topics(doc_bow)
    topics_prob = np.array(topics_prob)
    topic = np.argmax(topics_prob[:,1])
    return topic

def load_raw_data(dataset_dir, only_binary=True):
    """
        ids: list of tweet ids that are in tree file names list and has binary labels.
        contents: tweet text corresponding to ids.
    """
    trees_to_parse = get_tree_file_names(dataset_dir)
    labels = {}
    with open(os.path.join(dataset_dir, "label.txt")) as label_file:
        for line in label_file.readlines():
            label, news_id = line.split(":")
            labels[int(news_id)] = label
    news_ids_to_consider = list(labels.keys())
    if only_binary:
        news_ids_to_consider = [news_id for news_id in news_ids_to_consider
                                if labels[news_id] in ['false', 'true']]
    id_tweet_dict = load_tweet_content(dataset_dir+"/source_tweets.txt")
    ids = []
    contents = []
    for tree_file_name in trees_to_parse:
        news_id = get_root_id(tree_file_name)
        if news_id in news_ids_to_consider:
            ids.append(news_id)
            contents.append(id_tweet_dict[news_id])
    return ids, contents, labels


def load_tweet_content(src_file_name):
    """
        id_tweet_dict:{tweet_id->tweet_content}
    """
    id_tweet_dict = {}
    with open(src_file_name) as f:
        f_content = f.read()
    f_content = f_content.split('\n')
    for item in f_content:
        item = item.split('\t')
        if len(item)==2:
            id_tweet_dict[int(item[0])] = item[1]
    return id_tweet_dict

def gen_raw_tasks(ids, topics, labels, way_num, shot_num, seed):
    """
        tasks: list(topic_num, way_num*shot_num), contains the ids of tweets
    """
    ids_of_topics = []
    topics_list = list(set(topics))
    topics_list = sorted(topics_list)

    label_list = list(set([labels[id_] for id_ in labels]))
    if way_num == 2:
        label_list = ['true', 'false']
    print("Labels:{}".format(label_list))
    # tasks = np.zeros((len(topics_list), way_num*shot_num))
    topic_id_dict = {}
    for topic in topics_list:
        topic_id_dict[topic] = {label:[] for label in label_list}
    for idx in range(len(ids)):
        id_ = ids[idx]
        if labels[id_] in label_list:
            topic_id_dict[topics[idx]][labels[id_]].append(id_)
    # for topic in topics_list:
    #     for label in range(way_num):
    #         topic_id_dict[topic][label] = np.array(topic_id_dict[topic][label])

    print("Topics distribution:")
    topics_distr = []
    for topic in topic_id_dict:
        print([len(topic_id_dict[topic][label]) for label in label_list])
        topics_distr.append([len(topic_id_dict[topic][label]) for label in label_list])

    if (np.min(topics_distr)<shot_num):
        return [], False

    # for topic in topic_id_dict:
    #     topic_sample_num = 0
    #     for label in label_list:
    #         topic_sample_num += len(topic_id_dict[topic][label])
    #     if topic_sample_num < way_num*shot_num:
    #         return [], False

    np.random.seed(seed)
    rand_seeds = np.random.permutation(len(topics_list))

    raw_tasks = []
    for idx in range(len(topics_list)):
        topic = topics_list[idx]
        raw_task = []
        for label in label_list:
            raw_task += topic_id_dict[topic][label]
        raw_task_seed = rand_seeds[idx]
        np.random.seed(raw_task_seed)
        np.random.shuffle(raw_task)

        raw_tasks.append(raw_task)
    raw_tasks = np.array(raw_tasks)

    return raw_tasks, True

def gen_raw_tasks_by_label(ids, topics, labels, way_num, shot_num):
    """
        tasks: list(topic_num, way_num*shot_num), contains the ids of tweets
    """
    ids_of_topics = []
    topics_list = list(set(topics))
    topics_list = sorted(topics_list)

    label_list = list(set([labels[id_] for id_ in labels]))
    if way_num == 2:
        label_list = ['true', 'false']
    print("Labels:{}".format(label_list))
    # tasks = np.zeros((len(topics_list), way_num*shot_num))
    topic_id_dict = {}
    for topic in topics_list:
        topic_id_dict[topic] = {label:[] for label in label_list}
    for idx in range(len(ids)):
        id_ = ids[idx]
        if labels[id_] in label_list:
            topic_id_dict[topics[idx]][labels[id_]].append(id_)

    print("Topics distribution:")
    for topic in topic_id_dict:
        print([len(topic_id_dict[topic][label]) for label in label_list])

    for topic in topic_id_dict:
        topic_sample_num = 0
        for label in label_list:
            topic_sample_num += len(topic_id_dict[topic][label])
        if topic_sample_num < way_num*shot_num:
            return [], False

    raw_tasks_by_label = []
    for idx in range(len(topics_list)):
        topic = topics_list[idx]
        raw_task_by_label = {}
        for label in label_list:
            raw_task_by_label[label] = topic_id_dict[topic][label]
        raw_tasks_by_label.append(raw_task_by_label)

    return raw_tasks_by_label, True


def sample_task_from_raw_task(raw_task, support_shots, query_shots):
    np.random.seed()
    sampled_task = {}
    split_point = int(support_shots/(support_shots+query_shots)*len(raw_task))
    # print(split_point)
    # print(len(raw_task))
    # print(raw_task[:split_point])
    # print(raw_task[split_point:])
    sampled_task['s'] = np.random.choice(raw_task[:split_point], size=support_shots, \
        replace=True, p=None)
    sampled_task['q'] = np.random.choice(raw_task[split_point:], size=query_shots, \
        replace=True, p=None)
    # # no sample
    # sampled_task = {}
    # sampled_task['s'] = raw_task[:support_shots]
    # sampled_task['q'] = raw_task[support_shots:]

    return sampled_task

def sample_task_from_raw_task_by_label(raw_task_by_label, s_shot_per_class, q_shot_per_class, sample_flag=True):
    np.random.seed()
    label_list = [label for label in raw_task_by_label]
    if not sample_flag:
        # no sample
        sampled_task = {}
        sampled_task['s'] = []
        for label in label_list:
            sampled_task['s'] += list(raw_task_by_label[label][:s_shot_per_class])
        sampled_task['s'] = np.array(sampled_task['s'])
        np.random.shuffle(sampled_task['s'])

        sampled_task['q'] = []
        for label in label_list:
            sampled_task['q'] += list(raw_task_by_label[label][s_shot_per_class:])
        sampled_task['q'] = np.array(sampled_task['q'])
        np.random.shuffle(sampled_task['q'])
    else:
        sampled_task = {}
        split_points = {label:int(s_shot_per_class/(s_shot_per_class+q_shot_per_class)*len(raw_task_by_label[label])) \
            for label in label_list}
        # print(split_point)
        # print(len(raw_task))
        # print(raw_task[:split_point])
        # print(raw_task[split_point:])
        sampled_task['s'] = []
        for label in label_list:
            sampled_task['s'] += list(np.random.choice(raw_task_by_label[label][:split_points[label]], size=s_shot_per_class, \
                    replace=True, p=None))
        sampled_task['s'] = np.array(sampled_task['s'])
        np.random.shuffle(sampled_task['s'])

        sampled_task['q'] = []
        for label in label_list:
            sampled_task['q'] += list(np.random.choice(raw_task_by_label[label][split_points[label]:], size=q_shot_per_class, \
                    replace=True, p=None))
        sampled_task['q'] = np.array(sampled_task['q'])
        np.random.shuffle(sampled_task['q'])


    return sampled_task

def load_struct_data(tasks_ids, dataset):
    data = []
    for task_ids in tasks_ids:
        task_data = {'data_all':[], 'padded_docs':[], 'cos':[], 'label':[]}
        for news_id in task_ids:
            task_data['data_all'].append(dataset[news_id]['data_all'])
            task_data['padded_docs'].append(dataset[news_id]['padded_docs'])
            task_data['cos'].append(dataset[news_id]['cos'])
            task_data['label'].append(dataset[news_id]['label'])
        for key in task_data:
            task_data[key] = np.array(task_data[key])
        data.append(task_data)
    return data

def train_word2vec(sentences, model_file, word_dim, load_model=True):
    if not load_model or not os.path.exists(model_file):
        model = Word2Vec(sentences, sg=1, size=word_dim, window=6, min_count=1, negative=3, sample=0.001, hs=1, workers=1)
        model.save(model_file)
    else:
        #load model
        model = Word2Vec.load(model_file)
    return model


def gen_topics_embedding(corpus, num_topics, dictionary, word_dim, w2v_model):
    lda_model = train_lda(True, corpus, num_topics, dictionary)

    temp = dictionary[0] # This is only to "load" the dictionary.

    topics_emb = np.zeros((num_topics, word_dim))
    for topic_id in range(num_topics):
        topic = lda_model.get_topic_terms(topicid=topic_id, topn=10)
        topic_emb = np.zeros(word_dim)
        weights = []
        vecs = []
        for idx in range(len(topic)):
            word = dictionary.id2token[topic[idx][0]]
            weights.append(topic[idx][1])
            vecs.append(w2v_model[word])
        weights = np.array(weights)
        weights = weights/weights.sum()
        for idx in range(len(topic)):
            topic_emb += weights[idx]*vecs[idx]
        topics_emb[topic_id] = topic_emb
        # print(topic_emb)

    sim_mat = np.zeros((num_topics, num_topics))
    for i in range(num_topics):
        for j in range(num_topics):
            sim_mat[i][j] = cos_similarity(topics_emb[i], topics_emb[j])
    print("Topics similarity matrix:\n{}".format(sim_mat))
    return topics_emb




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Setup variables for MAML.')
    parser.add_argument('--datasource', type=str, default='twitter15', help='Datasource: twitter15, twitter16')
    parser.add_argument('--train_model', dest='load_model', action='store_false')
    parser.set_defaults(load_model=True)
    args = parser.parse_args()
    # --------------------------------------------------
    # Parse dataset and related variables
    # --------------------------------------------------
    datasource = args.datasource
    print('Dataset = {0:s}'.format(datasource))
    load_model = args.load_model
    # parameters setting
    num_topics = 10 #
    dataset_dir = os.path.join(DATA_DIR, datasource)
    way_num = 2
    shot_num = 8   # number of samples per class per topic
    seed = 10   #seed of random function
    sample_rate = 2
    
    support_shots = 12
    query_shots = 8


    ids, docs, labels = load_raw_data(dataset_dir)
    


    docs_text, corpus, dictionary = text_preprocess(docs, NO_BELOW, NO_ABOVE, MIN_COUNT)
    # tasks_ids: list of tasks
    # task_ids: list of tweet ids in the task, which contains way_num*shot_num samples
    
    if load_model and os.path.exists(MODEL_PATH):
        topics = gen_topics(corpus, num_topics, dictionary, load_model=load_model)
        raw_tasks, success_flag = gen_raw_tasks(np.array(ids), topics, labels, way_num, shot_num, seed)
        tasks = []
        for raw_task in raw_tasks:
            task = sample_task_from_raw_task(raw_task, support_shots, query_shots)
            tasks.append(task)
    else:
        success_flag = False
        max_gen_times = 20  
        for i in range(max_gen_times):
            topics = gen_topics(corpus, num_topics, dictionary, load_model=load_model)
            raw_tasks, success_flag = gen_raw_tasks(np.array(ids), topics, labels, way_num, shot_num, seed)
            tasks = []
            for raw_task in raw_tasks:
                task = sample_task_from_raw_task(raw_task, support_shots, query_shots)
                tasks.append(task)
            if success_flag:
                break
        if not success_flag:
            print("Failed to generate valid tasks.")
        else:
            print("Successed to generate valid tasks.")
    w2v_model = train_word2vec(docs_text, WORD2VEC_MODEL_PATH, WORD_DIM, load_model=True)
    topics_emb = gen_topics_embedding(corpus, num_topics, dictionary, WORD_DIM, w2v_model)

    
