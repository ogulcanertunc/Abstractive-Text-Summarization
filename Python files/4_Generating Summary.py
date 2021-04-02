!pip install transformers
import transformers
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, DistilBertModel, DistilBertTokenizer, BertTokenizer, BertForTokenClassification
import numpy as np

import nltk
nltk.download('punkt')
from nltk import sent_tokenize
%tensorflow_version 1.x
from keras.preprocessing.sequence import pad_sequences

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

import json
import matplotlib.pyplot as plt
import timeit
import torch
import textwrap
wrapper = textwrap.TextWrapper(width=70)
SEED = 1234
torch.manual_seed(SEED)

### 1. Determine the Model Directory ###
########################################


GPT2_config_directory = '/content/GPT2_dir/GPT2_dir/trained_model'
print('where is GPT2 dir ? ',GPT2_config_directory)

### 1.1 Assign pre-trained models from the Directory ###
########################################################

tokenizer_GPT2 = GPT2Tokenizer.from_pretrained(
                  GPT2_config_directory)
special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}
tokenizer_GPT2.add_special_tokens(special_tokens)
GPT2_generator = GPT2DoubleHeadsModel.from_pretrained(
                  GPT2_config_directory)

print('----loading GPT2 summary generator----')
tokenizer_GPT2 = GPT2Tokenizer.from_pretrained(GPT2_config_directory)
special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}
tokenizer_GPT2.add_special_tokens(special_tokens)
GPT2_generator = GPT2DoubleHeadsModel.from_pretrained(GPT2_config_directory)
print('----GPT2 summary generator directory is true----')

### 1.2 Activate/Deactivate GPU ###
use_GPU_GPT_generator = True

if torch.cuda.is_available():
  print('cuda is available')
  device = 'cuda'
  print('device is set to cuda')
if not torch.cuda.is_available():
  print('cuda is not available')
  device = 'cpu'
  print('device is set to cpu')
  use_GPU_GPT_generator = False

print('use GPU for GPT2?' ,use_GPU_GPT_generator)

### 1.3 Create a Keywords Maker ###

import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def get_keywords(text_):
    some_text = text_
    lowered = some_text.lower()
    tokens = nltk.tokenize.word_tokenize(lowered)
    keywords = [keyword for keyword in tokens if keyword.isalpha() and not keyword in stop_words]
    keywords_string = ','.join(keywords)
    return keywords_string


### 2. Data Entry ###
#####################

text = "Despite making pejorative remarks about Africa, US President Donald Trump has attracted a devout following " \
       "among some Christians on the continent.  Pray for him [Trump] because when God places any of his children " \
       "in a position, hell sometimes would do everything to destroy that individual, said Nigerian Pastor Chris" \
       " Oyakhilome, a prominent televangelist, in a sermon in June.  He has also warned that critics of the " \
       "Republican president, who is seeking re-election in November, dislike his supporters.  They are angry at Trump" \
       " for supporting Christians, you better know it. So the real ones that they hate are you who are Christians," \
       " said the pastor, whose broadcasts are popular around the world, including in the US.  President Trump " \
       "has been a polarising figure the world over but he is popular in African countries like Nigeria and Kenya, " \
       "according to a Pew Research poll released in January, where supporters do not appear to be bothered that he " \
       "reportedly referred to African countries as shitholes in 2018.  Both Nigeria and Kenya are deeply religious " \
       "countries. Mega churches proliferate in the Christian south of Nigeria - Africa's most populous nation - and in " \
       "Kenya many politicians go to church sermons to address their supporters, such is their popularity"

automated_keywords   = True
if automated_keywords == True:
  list_keywords = get_keywords(text)
  print('keywords are generated automatically')
if not automated_keywords:
  list_keywords = 'If automated keywords is not selected pls enter the list_keywords as a string'

title = 'US elections: The African evangelicals praying for Trump to win'

print('It is ready to summarizing')


GPT2_input = tokenizer_GPT2.encode(
      '<|startoftext|> ' +title + list_keywords + ' <|summarize|> ')
GPT2_input_torch = torch.tensor(GPT2_input, dtype=torch.long)

print("the keyword input :")
wrapper.wrap(tokenizer_GPT2.decode(GPT2_input_torch))


wrapper.wrap(title+list_keywords)

print (GPT2_input_torch)

# this step may takes a few mins without GPU
temperature = 1
greedy_search = False
top_k = 35
top_p = 0.8
max_length = 200

min_length = 20
num_return_sequences = 3

if use_GPU_GPT_generator:
    GPT2_generator = GPT2_generator.to(device)
    GPT2_input_torch = GPT2_input_torch.to(device)

do_sample = not greedy_search
if do_sample == False:
    num_return_sequences = 1

sampling_output = GPT2_generator.generate(
    input_ids=GPT2_input_torch.unsqueeze(0),
    max_length=max_length + len(GPT2_input_torch),
    min_length=min_length + len(GPT2_input_torch),
    temperature=temperature,
    decoder_start_token_id='<|summarize|>',
    top_k=top_k,
    top_p=top_p,
    do_sample=do_sample,
    num_return_sequences=num_return_sequences,
    no_repeat_ngram_size=3)

print('finish generating')




which_output = 2
wrapper.wrap(tokenizer_GPT2.decode(
    sampling_output[which_output,len(GPT2_input_torch):],
    skip_special_tokens=True)[:5000])