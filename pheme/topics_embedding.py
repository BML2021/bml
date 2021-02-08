import numpy as np

from abml_GCAN import data_scaling
from utils import load_raw_dataset, gen_topics_embedding
from cfg import *

def get_topic_index(raw_dataset):
    topic_index = {}
    for topic in raw_dataset.keys():
        topic_index[topic] = {0:[], 1:[]}
        for raw_news in raw_dataset[topic]['non-rumours']:
            topic_index[topic][1].append(raw_news['tweet_id'])
            # break # test
        for raw_news in raw_dataset[topic]['rumours']:
            topic_index[topic][0].append(raw_news['tweet_id'])
    return topic_index

if __name__ == '__main__':
    # load raw dataset
    raw_dataset = load_raw_dataset(DATA_ROOT_PATH)
    topic_index = get_topic_index(raw_dataset)

    # raw_task: [[0:[t_ids],1:[t_ids]],...]
    raw_tasks = [topic_index[topic] for topic in topic_index.keys()]

    data_scale = 1.0
    #scale data in tasks
    raw_tasks = data_scaling(raw_tasks, data_scale)
    print("scaled task distribution:")
    for task in raw_tasks:
        print([len(task[key]) for key in task.keys()])

    # calculate topic embedding
    topics_emb = gen_topics_embedding(raw_dataset, raw_tasks)
    np.save('topics_emb.npy', topics_emb)