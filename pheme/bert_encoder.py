#tensorflow.__version__=1.13.1
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import torch
import numpy as np
import warnings
from dataset import load_tweet_ids_with_text, load_raw_dataset
from cfg import DATA_ROOT_PATH
import time
from utils import gen_topics_embedding

if __name__ == '__main__':	
	warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
	start = time.time()
	tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
	model = XLMRobertaModel.from_pretrained('xlm-roberta-base')

	raw_dataset = load_raw_dataset(DATA_ROOT_PATH)
	id_texts = load_tweet_ids_with_text(raw_dataset)
	ids = [id_texts[idx][0] for idx in range(len(id_texts))]
	texts = [id_texts[idx][1] for idx in range(len(id_texts))]
	print('amount of news: {}'.format(len(texts)))
	
	result = []
	for i in range(len(texts)):
		input_ids = torch.tensor([tokenizer.encode(texts[i], add_special_tokens=True)])[:,:512]
		
		with torch.no_grad():
		    outputs = model(input_ids)[1].tolist()
		result.append(outputs)
	np.save('output_bert.npy', result)
	np.save('input_t_id.npy', ids)

	print('Finished in {}s'.format(time.time()-start))
