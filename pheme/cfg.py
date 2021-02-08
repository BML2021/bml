import os

# path parameters setting
DATA_ROOT_PATH = '../PHEME_veracity/all-rnr-annotated-threads'

if not os.path.exists("model/inner_model"):
    os.makedirs("model/inner_model")

# text preprocess parameters
NO_BELOW = 5
NO_ABOVE = 0.1
MIN_COUNT = 20

# dataset parameter
USER_FTS_DIM = 9
# BERT_EMB_DIM = TEXT_LEN*SOURCE_TWEET_OUTPUT_DIM
TEXT_LEN = 24
SOURCE_TWEET_OUTPUT_DIM = 32
BERT_EMB_DIM = 768

#-------------------------
# Model parameters setting
#-------------------------
OUTPUT_DIM = 64 
GCN_FILTERS_NUM = 1
GCN_OUTPUT_DIM = 32
CNN_FILTER_SIZE = (3,3)
CNN_OUTPUT_DIM = 32
STRIDE_SIZE = 1
MIN_EPOCH_NUM = 50
MAX_EPOCH_NUM = 100