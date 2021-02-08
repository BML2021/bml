# dataset setting: binary, graph
import numpy as np
import os
import io
import sys
import random
import time
from collections import defaultdict
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

# from utils import get_tree_file_names, get_root_id, parse_edge_line, from_date_text_to_timestamp, to_label
from cfg import DATA_ROOT_PATH
from utils import from_date_text_to_timestamp, load_tweet_ids_with_text, load_raw_dataset

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8') 

class DatasetBuilder:

    def __init__(self, raw_dataset, retweet_user_size = 30, seed=64):
        self.seed = seed
        self.raw_dataset = raw_dataset
        self.retweet_user_size = retweet_user_size
        self.only_binary = True # to be deleted
        self.time_cut = None # to be deleted
        
    def create_dataset(self):
        """
        Args:
            dataset_type:str. Has to be 'topic_index', 'id_index'
        Returns:
            dict with topic keys:
        """

        # if dataset_type not in ['topic_index', 'id_index']:
        #     raise ValueError("Supported dataset types are: 'topic_index', 'id_index'.")

        start_time = time.time()
        
        tweet_features = self.load_tweet_features_bert(self.raw_dataset)
        user_features = self.load_user_features(self.raw_dataset)
        # show user features
        print("User features:")
        for key in user_features:
            for k in user_features[key]:
                print('\t'+k)
            break

        # if dataset_type == 'topic_index':
        # 0:fake news, 1:true news
        dataset = {}
        topic_index = {}
        for topic in self.raw_dataset.keys():
            topic_index[topic] = {0:[], 1:[]}
            for raw_news in self.raw_dataset[topic]['non-rumours']:
                news = self.preprocess_news(raw_news, tweet_features, user_features)
                dataset[news['id']] = news
                topic_index[topic][1].append(news['id'])
                # break # test
            for raw_news in self.raw_dataset[topic]['rumours']:
                news = self.preprocess_news(raw_news, tweet_features, user_features)
                dataset[news['id']] = news
                topic_index[topic][0].append(news['id'])
                    
        print(f"Dataset loaded in {time.time() - start_time:.3f}s")
        return dataset, topic_index
    def preprocess_news(self, raw_news, tweet_features, user_features):
        # print(raw_news.keys())
        # print(raw_news['source_tweet'].keys())
        # print(len(raw_news['reactions']))
        news = {}
        if raw_news['annotation']['is_rumour'] == 'nonrumour':
            news['label'] = 1
        else:
            news['label'] = 0
        news['id'] = raw_news['tweet_id']
        news['padded_docs'] = tweet_features[news['id']]
        retweeters = self.get_retweeters(raw_news['reactions'], raw_news['source_tweet'])
        retweet_dists = self.get_retweeter_dists(raw_news['reactions'], \
            raw_news['structure'], raw_news['source_tweet'])
        adj, retweeter_fts = self.get_retweeter_adj(retweeters, retweet_dists, \
            user_features)
        retweeter_fts.astype('float')
        news['data_all'] = retweeter_fts
        news['cos'] = adj
        return news

    def load_tweet_features_bert(self, raw_dataset):
        """
        Text features embeded by BERT
        Returns:
            tweet_texts: dict[tweet_id:int -> dict[name feature -> feature]]
        """
        tweet_features = {}
        text_embeddings = np.load("output_bert.npy") #[tweet feature embeddings]
        text_embeddings = np.squeeze(text_embeddings)  
        tweet_features_id = np.load("input_t_id.npy") #[tweet ids]
        # id_texts = load_tweet_ids_with_text(raw_dataset)
        # tweet_features_id = [item[0] for item in id_texts]
        for i, tweet_id in enumerate(tweet_features_id):
            tweet_features[int(tweet_id)] = text_embeddings[i]
        return tweet_features

    def load_user_features(self, raw_dataset):
        """
        Returns:
            user_features: dict[user_id:int -> dict[name feature -> feature]]
        """
        t1 = time.time()
        user_features = {}
        for topic in raw_dataset.keys():
            for key in raw_dataset[topic].keys():
                for tweet in raw_dataset[topic][key]:
                    tweet_list = [tweet['source_tweet']]+tweet['reactions']
                    for source_tweet in tweet_list:
                        raw_user_fts = source_tweet['user']
                        user_id = raw_user_fts['id']
                        if user_id not in user_features.keys():
                            user_features[user_id] = {}
                            user_features[user_id]['created_at'] = float(from_date_text_to_timestamp(raw_user_fts['created_at']))
                            user_features[user_id]['favourites_count'] = float(raw_user_fts['favourites_count'])
                            user_features[user_id]['followers_count'] = float(raw_user_fts['followers_count'])
                            user_features[user_id]['friends_count'] = float(raw_user_fts['friends_count'])
                            if raw_user_fts['geo_enabled']:
                                user_features[user_id]['geo_enabled'] = 1.0
                            else:
                                user_features[user_id]['geo_enabled'] = 0.0
                            if 'description' not in raw_user_fts.keys() or \
                                raw_user_fts['description'] == '':
                                user_features[user_id]['has_description'] = 0.0
                            else:
                                user_features[user_id]['has_description'] = 1.0
                            user_features[user_id]['len_name'] = float(len(raw_user_fts['name']))
                            user_features[user_id]['len_screen_name'] = float(len(raw_user_fts['screen_name']))
                            user_features[user_id]['listed_count'] = float(raw_user_fts['listed_count'])
                            user_features[user_id]['statuses_count'] = float(raw_user_fts['statuses_count'])
                            if raw_user_fts['verified']:
                                user_features[user_id]['verified'] = 1.0
                            else:
                                user_features[user_id]['verified'] = 0.0

        # # user_features_train_only = {key: val for key, val in user_features.items() if key in user_ids_in_train}
        # print("Standardizing features")
        # # Standardizing
        # for ft in ["created_at", "favourites_count", "followers_count", \
        #     "friends_count", "listed_count", "statuses_count"]:
        #     scaler = StandardScaler().fit(
        #         np.array([val[ft] for val in user_features.values()]).reshape(-1, 1)
        #     )

        #     # faster to do this way as we don't have to convert to np arrays
        #     mean, std = scaler.mean_[0], scaler.var_[0] ** (1 / 2)
        #     for key in user_features.keys():
        #         user_features[key][ft] = (user_features[key][ft] - mean) / std

        #     # user_features_train_only = {key: val for key, val in user_features.items() if key in user_ids_in_train}

        # dict_defaults = {
        #     'created_at': np.median([elt["created_at"] for elt in user_features.values()]),
        #     'favourites_count': np.median([elt["favourites_count"] for elt in user_features.values()]),
        #     'followers_count': np.median([elt["followers_count"] for elt in user_features.values()]),
        #     'friends_count': np.median([elt["friends_count"] for elt in user_features.values()]),
        #     'geo_enabled': 0,
        #     'has_description': 0,
        #     'len_name': np.median([elt["len_name"] for elt in user_features.values()]),
        #     'len_screen_name': np.median([elt["len_screen_name"] for elt in user_features.values()]),
        #     'listed_count': np.median([elt["listed_count"] for elt in user_features.values()]),
        #     'statuses_count': np.median([elt["statuses_count"] for elt in user_features.values()]),
        #     'verified': 0
        # }

        # def default_user_features():
        #     """ Return np array of default features sorted by alphabetic order """
        #     return np.array([val for key, val in
        #                      sorted(dict_defaults.items(), key=lambda x: x[0])])

        # #  user features: key=uid, value=dict[ftname:valueft]
        # np_user_features = {key: np.array([key_val[1] for key_val in sorted(value.items(), key=lambda x: x[0])]) for
        #                     key, value in user_features.items()}

        t2 = time.time()
        print('load user features from raw dataset: {}s'.format(t2-t1))
        # return defaultdict(default_user_features, np_user_features)
        return user_features

    
    def get_retweeter_adj(self, retweeters, retweet_dists, user_features):
        # Build users' features in format of numpy array
        u_fts_mat = []
        for retweeter in retweeters:
            discri_num = user_features[retweeter]['has_description']
            screen_name_num = user_features[retweeter]['len_screen_name']
            followers_num = user_features[retweeter]['followers_count']
            favourites_num = user_features[retweeter]['favourites_count']
            stories_num = user_features[retweeter]['statuses_count']
            time_elapsed = user_features[retweeter]['created_at']
            verified_state = user_features[retweeter]['verified']
            geo_setting = user_features[retweeter]['geo_enabled']
            # time_out = time_outs[retweeter]
            retweet_dist = retweet_dists[retweeter]
            u_fts_mat.append([discri_num, screen_name_num, followers_num, favourites_num,
                stories_num, time_elapsed, verified_state, geo_setting, retweet_dist])
        u_fts_mat = np.array(u_fts_mat)
        # print("min:{} max:{}".format(np.min(u_fts_mat), np.max(u_fts_mat)))
        # Standardizing
        # print("Standardizing...")
        # print(u_fts_mat.shape)
        for i in range(u_fts_mat.shape[1]):
            scaler = StandardScaler().fit(
                u_fts_mat[:,i].reshape(-1, 1)
            )
            # faster to do this way as we don't have to convert to np arrays
            mean, std = scaler.mean_[0], scaler.var_[0] ** (1 / 2)
            # print("mean:{} std:{}".format(mean, std))
            for j in range(u_fts_mat.shape[0]):
                if std != 0:
                    u_fts_mat[j][i] = (u_fts_mat[j][i] - mean) / std
                else:
                    u_fts_mat[j][i] = (u_fts_mat[j][i] - mean)
        # print("min:{} max:{}".format(np.min(u_fts_mat), np.max(u_fts_mat)))

        # Compute the similarity matrix of retweeter queue
        adj = cosine_similarity(u_fts_mat, u_fts_mat)
        return adj, u_fts_mat

    def get_retweeters(self, reactions, source_tweet):
        retweeters = []
        for reaction in reactions:
            retweeters.append(reaction['user']['id'])
        if (len(retweeters) >= self.retweet_user_size):
            return retweeters[:self.retweet_user_size]
        else:
            # if retweet users less than the fixed number, 
            # pad by random sample from current retweet users
            if len(retweeters) == 0:
                return [source_tweet['user']['id'] for _ in range(self.retweet_user_size)]
            else:
                src_retweeters = np.array(retweeters)
                q = src_retweeters[np.random.choice(src_retweeters.shape[0], \
                    self.retweet_user_size, replace=True)]
                q = q.tolist()
                src_retweeters = src_retweeters.tolist()
                src_retweeters.extend(q)
                retweeters = src_retweeters[:self.retweet_user_size]
                return retweeters

    def get_retweeter_dists(self, reactions, structure, source_tweet):
        reactions_dict = {tweet['id']:tweet for tweet in reactions}
        reactions_dict[source_tweet['id']] = source_tweet
        
        lens_dict = {}
        depth = 0
        struct = structure.copy()
        nodes = [(struct, struct.keys())]
        while (len(nodes) > 0):
            new_nodes = []
            for node, node_keys in nodes:
                for node_key in node_keys:
                    # print(int(node_key), reactions_dict[int(node_key)]['user']['id'])
                    lens_dict[reactions_dict[int(node_key)]['user']['id']] = depth
                    if node[node_key] != []:
                        new_nodes += [(node[node_key], node[node_key].keys())]
            depth += 1
            nodes = new_nodes
        return lens_dict

if __name__ == '__main__':
    # load raw dataset
    raw_dataset = load_raw_dataset(DATA_ROOT_PATH)
    print(raw_dataset.keys())

    # build dataset with preprocessing
    # parameter setting
    retweet_user_size = 30
    data_builder = DatasetBuilder(raw_dataset, retweet_user_size)
    dataset, topic_index = data_builder.create_dataset()
    print('Topics in dataset: {}'.format(dataset.keys()))
    print('Dataset size: {}'.format(len(dataset)))

    # dataset2 = data_builder.create_dataset(dataset_type="id_index", standardize_features=True)
    # np.set_printoptions(threshold=1e6)
    # if not os.path.exists(DATA_DIR+"/"+datasource+"/topic_index"):
    #     os.makedirs(DATA_DIR+"/"+datasource+"/topic_index")
    # with open(DATA_DIR+"/"+datasource+"/topic_index/processed_dataset_{}.txt".format(retweet_user_size), "w") as f:
    #     f.write(str(dataset1))
    # if not os.path.exists(DATA_DIR+"/"+datasource+"/id_index"):
    #     os.makedirs(DATA_DIR+"/"+datasource+"/id_index")
    # with open(DATA_DIR+"/"+datasource+"/id_index/processed_dataset_{}.txt".format(retweet_user_size), "w") as f:
    #     f.write(str(dataset2))

