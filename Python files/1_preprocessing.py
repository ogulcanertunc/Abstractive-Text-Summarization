### 1. Import Libraries and datasets ###
########################################
import pandas as pd
import numpy as np
import timeit
import re

import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import nltk
nltk.download('punkt')

import textblob
from textblob import TextBlob



import torch
print(torch.__version__,' pytorch version')
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

!pip install transformers==2.6.0


import transformers
print(transformers.__version__,' make sure transformers version is 2.6.0')
from transformers import GPT2Tokenizer


### 1.2 Tokenizer and datasets ###
##################################

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}
tokenizer.add_special_tokens(special_tokens)
print(len(tokenizer), 'total length of vocab')
print(tokenizer.bos_token_id, 'bos_token')
print(tokenizer.eos_token_id, 'eos_token')
print(tokenizer.pad_token_id, 'pad_token')  #token for <pad>, len of all tokens in the tokenizer
print(tokenizer.additional_special_tokens_ids[0], 'keyword_token') #token for <|keyword|>
print(tokenizer.additional_special_tokens_ids[1], 'summary_token') #token for <|summarize|>


df_train = pd.read_csv('train_last_edt.csv',index_col=0)
df_test  = pd.read_csv('test_last_edt.csv',index_col=0)

#df_train.head()

### 2. Generate keywords ###
############################

def get_keywords(row):
    some_text = row['for_keyword']
    lowered = some_text.lower()
    tokens = nltk.tokenize.word_tokenize(lowered)
    keywords = [keyword for keyword in tokens if keyword.isalpha() and not keyword in stop_words]
    keywords_string = ','.join(keywords)
    return keywords_string


def pre_process_ogi(df):
    deneme = df
    deneme['for_keyword'] = deneme['headlines'] + deneme['ctext']
    deneme['keywords'] = deneme.apply(get_keywords, axis=1)
    deneme['keyword_POS'] = deneme["keywords"].apply(lambda x: TextBlob(x).words)
    deneme["keyword_POS_str"] = deneme.keyword_POS.apply(', '.join)
    deneme['keyword_POS_str_nocomma'] = deneme.keyword_POS.apply(' '.join)

    return deneme

train_process = pre_process_ogi(df_train)
test_process = pre_process_ogi(df_test)






















































